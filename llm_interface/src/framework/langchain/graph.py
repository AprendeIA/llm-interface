import operator
from typing import Dict, List, Any, Optional, Callable, Tuple
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import Runnable
from typing import TypedDict, Annotated
from ...manager import LLMManager

class GraphState(TypedDict):
    """Base state for LangGraph workflows
    
    Attributes:
        messages: List of messages in the conversation (accumulative)
        provider: Name of the provider to use for this request
        response: The response from the LLM
        context: Additional context data for the workflow
    """
    messages: Annotated[list, operator.add]
    provider: str
    response: str
    context: Dict[str, Any]

class LLMGraph:
    """Main LangGraph wrapper for LLM providers"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.graph = None
        self.custom_nodes: Dict[str, Callable] = {}
        self.custom_edges: Dict[str, Dict[str, str]] = {}
    
    def create_simple_chat_graph(self) -> StateGraph:
        """Create a simple chat graph with provider routing
        
        Returns:
            StateGraph: A compiled graph for simple chat interactions
        """
        workflow = StateGraph(GraphState)
        
        # Add processing nodes for each provider
        provider_names = list(self.llm_manager.list_providers())
        if not provider_names:
            raise ValueError("No providers available. Add at least one provider to the manager first.")
        
        for provider_name in provider_names:
            node_name = f"process_{provider_name}"
            workflow.add_node(node_name, self._create_chat_node(provider_name))
        
        # Add conditional routing with mapping
        routing_map = {
            f"process_{name}": f"process_{name}"
            for name in provider_names
        }
        
        workflow.set_conditional_entry_point(
            self._route_to_provider,
            routing_map
        )
        
        # Connect all processing nodes to end
        for provider_name in provider_names:
            workflow.add_edge(f"process_{provider_name}", END)
        
        return workflow
    
    def _create_chat_node(self, provider_name: str) -> Callable[[GraphState], Dict[str, Any]]:
        """Create a chat processing node for a specific provider
        
        Args:
            provider_name: Name of the provider to use
            
        Returns:
            Callable: A function that processes chat messages through the provider
        """
        def chat_node(state: GraphState) -> Dict[str, Any]:
            try:
                model = self.llm_manager.get_chat_model(provider_name)
                response = model.invoke(state["messages"])
                
                return {
                    "response": response.content if hasattr(response, 'content') else str(response),
                    "messages": [AIMessage(content=response.content if hasattr(response, 'content') else str(response))]
                }
            except Exception as e:
                return {
                    "response": f"Error with {provider_name}: {str(e)}",
                    "messages": [AIMessage(content=f"Error: {str(e)}")]
                }
        
        return chat_node
    
    def _route_to_provider(self, state: GraphState) -> str:
        """Route to the appropriate provider based on state
        
        Args:
            state: Current graph state with provider information
            
        Returns:
            str: Node name to route to
            
        Raises:
            ValueError: If no providers are available
        """
        
        provider = state.get("provider", "")
        
        if provider in self.llm_manager.list_providers():
            return f"process_{provider}"
        
        # Default to first available provider
        first_provider = next(iter(self.llm_manager.list_providers()), None)
        
        if first_provider:
            return f"process_{first_provider}"
        
        raise ValueError("No providers available")
    
    def add_custom_node(self, name: str, node_function: Callable[[GraphState], Dict[str, Any]], 
                       input_state_type: Optional[type] = None) -> None:
        """Add a custom node to the graph
        
        Args:
            name: Name of the node
            node_function: Function that processes the graph state
            input_state_type: Optional custom state type
        """
        self.custom_nodes[name] = {
            'function': node_function,
            'input_type': input_state_type or GraphState
        }
    
    def add_conditional_node(self, name: str, condition_function: Callable[[GraphState], str],
                           outcomes: Dict[str, str]) -> None:
        """Add a conditional node with multiple outcomes
        
        Args:
            name: Name of the node
            condition_function: Function that determines routing based on state
            outcomes: Mapping of condition results to target nodes
        """
        self.custom_nodes[name] = {
            'function': condition_function,
            'outcomes': outcomes,
            'conditional': True
        }
    
    def build_custom_graph(self, workflow_definition: Dict[str, Any]) -> StateGraph:
        """Build a custom graph from workflow definition
        
        Args:
            workflow_definition: Dictionary containing nodes, edges, and entry point
            
        Returns:
            StateGraph: The constructed workflow graph
        """
        workflow = StateGraph(workflow_definition.get('state_type', GraphState))
        
        # Add nodes
        for node_name, node_config in workflow_definition.get('nodes', {}).items():
            if 'provider' in node_config:
                # LLM processing node
                workflow.add_node(node_name, self._create_chat_node(node_config['provider']))
            elif 'function' in node_config:
                # Custom function node
                workflow.add_node(node_name, node_config['function'])
        
        # Add edges
        edges = workflow_definition.get('edges', {})
        
        for source, target in edges.items():
            if isinstance(target, dict):
                # Conditional edge
                workflow.add_conditional_edges(source, target['condition'], target['mapping'])
            else:
                # Simple edge
                workflow.add_edge(source, target)
        
        # Set entry point
        if 'entry_point' in workflow_definition:
            workflow.set_entry_point(workflow_definition['entry_point'])
        
        return workflow
    
    def compile_graph(self, workflow: StateGraph) -> Runnable:
        """Compile the workflow into a runnable graph
        
        Args:
            workflow: The StateGraph to compile
            
        Returns:
            Runnable: Compiled and executable graph
        """
        return workflow.compile()
    
    def run_simple_chat(self, messages: List[BaseMessage], provider: str) -> Dict[str, Any]:
        """Run a simple chat interaction
        
        Args:
            messages: List of messages to process
            provider: Name of the provider to use
            
        Returns:
            Dict containing the response and updated state
        """
        if not self.graph:
            workflow = self.create_simple_chat_graph()
            self.graph = self.compile_graph(workflow)
        
        initial_state = {
            "messages": messages,
            "provider": provider,
            "response": "",
            "context": {}
        }
        
        return self.graph.invoke(initial_state)
    
    def run_with_context(self, messages: List[BaseMessage], provider: str, 
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run with additional context
        
        Args:
            messages: List of messages to process
            provider: Name of the provider to use
            context: Optional context dictionary
            
        Returns:
            Dict containing the response and updated state
        """
        if not self.graph:
            workflow = self.create_simple_chat_graph()
            self.graph = self.compile_graph(workflow)
        
        initial_state = {
            "messages": messages,
            "provider": provider,
            "response": "",
            "context": context or {}
        }
        
        return self.graph.invoke(initial_state)

class MultiProviderRouter:
    """Advanced router for multiple providers with fallback logic
    
    This router manages provider selection with automatic fallback chains,
    allowing for graceful degradation when preferred providers are unavailable.
    """
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.fallback_chain = []  # Provider fallback order
    
    def set_fallback_chain(self, providers: List[str]) -> None:
        """Set the fallback chain for providers
        
        Args:
            providers: Ordered list of provider names for fallback routing
        """
        available_providers = set(self.llm_manager.list_providers())
        self.fallback_chain = [p for p in providers if p in available_providers]
    
    def route_with_fallback(self, state: GraphState) -> str:
        """Route with fallback logic
        
        Args:
            state: Current graph state with provider preference
            
        Returns:
            str: Node name to route to
            
        Raises:
            ValueError: If no providers are available
        """
        preferred_provider = state.get("provider", "")
        
        # Try preferred provider first
        if preferred_provider in self.llm_manager.list_providers():
            return f"process_{preferred_provider}"
        
        # Try fallback chain
        for provider in self.fallback_chain:
            if provider in self.llm_manager.list_providers():
                return f"process_{provider}"
        
        # Default to any available provider
        first_available = next(iter(self.llm_manager.list_providers()), None)
        if first_available:
            return f"process_{first_available}"
        
        raise ValueError("No providers available")

class GraphBuilder:
    """Builder pattern for creating complex graphs
    
    Provides a fluent interface for constructing LangGraph workflows
    with multiple nodes, edges, and providers.
    """
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.workflow = StateGraph(GraphState)
        self.nodes = {}
        self.edges = {}
    
    def add_llm_node(self, name: str, provider: str) -> 'GraphBuilder':
        """Add an LLM processing node
        
        Args:
            name: Node identifier
            provider: Provider name to use for this node
            
        Returns:
            GraphBuilder: Self for method chaining
        """
        def llm_node(state: GraphState) -> Dict[str, Any]:
            model = self.llm_manager.get_chat_model(provider)
            response = model.invoke(state["messages"])
            return {
                "response": response.content,
                "messages": state["messages"] + [AIMessage(content=response.content)]
            }
        
        self.workflow.add_node(name, llm_node)
        return self
    
    def add_custom_node(self, name: str, function: Callable[[GraphState], Dict[str, Any]]) -> 'GraphBuilder':
        """Add a custom processing node
        
        Args:
            name: Node identifier
            function: Function that processes graph state
            
        Returns:
            GraphBuilder: Self for method chaining
        """
        self.workflow.add_node(name, function)
        return self
    
    def set_entry_point(self, node_name: str) -> 'GraphBuilder':
        """Set the entry point
        
        Args:
            node_name: Name of the starting node
            
        Returns:
            GraphBuilder: Self for method chaining
        """
        self.workflow.set_entry_point(node_name)
        return self
    
    def add_edge(self, source: str, target: str) -> 'GraphBuilder':
        """Add a simple edge
        
        Args:
            source: Source node name
            target: Target node name
            
        Returns:
            GraphBuilder: Self for method chaining
        """
        self.workflow.add_edge(source, target)
        return self
    
    def add_conditional_edges(self, source: str, condition: Callable[[GraphState], str], 
                            mapping: Dict[str, str]) -> 'GraphBuilder':
        """Add conditional edges
        
        Args:
            source: Source node name
            condition: Function that determines next node based on state
            mapping: Dictionary mapping condition results to target nodes
            
        Returns:
            GraphBuilder: Self for method chaining
        """
        self.workflow.add_conditional_edges(source, condition, mapping)
        return self
    
    def build(self) -> Runnable:
        """Build and compile the graph
        
        Returns:
            Runnable: Compiled and executable graph
        """
        return self.workflow.compile()

# Utility functions for common graph patterns
def create_fallback_graph(llm_manager: LLMManager, 
                         providers: List[str]) -> Runnable:
    """Create a graph with automatic fallback between providers
    
    Args:
        llm_manager: The LLM manager with configured providers
        providers: List of provider names in fallback order
        
    Returns:
        Runnable: Compiled graph with fallback logic
    """
    workflow = StateGraph(GraphState)
    
    # Add nodes for each provider
    for provider in providers:
        if provider in llm_manager.list_providers():
            def create_provider_node(provider_name: str):
                def node(state: GraphState) -> Dict[str, Any]:
                    try:
                        model = llm_manager.get_chat_model(provider_name)
                        response = model.invoke(state["messages"])
                        return {
                            "response": response.content,
                            "provider_used": provider_name,
                            "success": True
                        }
                    except Exception as e:
                        return {
                            "response": "",
                            "provider_used": provider_name,
                            "success": False,
                            "error": str(e)
                        }
                return node
            
            workflow.add_node(f"try_{provider}", create_provider_node(provider))
    
    # Add routing logic
    def route_next_provider(state: GraphState) -> str:
        # This would contain logic to determine next provider
        # based on previous failures, etc.
        pass
    
    return workflow.compile()

def create_parallel_provider_graph(llm_manager: LLMManager, 
                                 providers: List[str]) -> Runnable:
    """Create a graph that queries multiple providers in parallel
    
    Args:
        llm_manager: The LLM manager with configured providers
        providers: List of provider names to query in parallel
        
    Returns:
        Runnable: Compiled graph with parallel processing
    """
    workflow = StateGraph(GraphState)
    
    # Add parallel processing nodes
    for provider in providers:
        if provider in llm_manager.list_providers():
            def create_parallel_node(provider_name: str):
                def node(state: GraphState) -> Dict[str, Any]:
                    model = llm_manager.get_chat_model(provider_name)
                    response = model.invoke(state["messages"])
                    return {f"response_{provider_name}": response.content}
                return node
            
            workflow.add_node(f"process_{provider}", create_parallel_node(provider))
    
    # Add aggregation node
    def aggregate_responses(state: GraphState) -> Dict[str, Any]:
        responses = {}
        for key, value in state.items():
            if key.startswith("response_"):
                responses[key] = value
        return {"aggregated_responses": responses}
    
    workflow.add_node("aggregate", aggregate_responses)
    
    # Connect all processing nodes to aggregator
    for provider in providers:
        if provider in llm_manager.list_providers():
            workflow.add_edge(f"process_{provider}", "aggregate")
    
    workflow.set_entry_point(f"process_{providers[0]}")
    workflow.add_edge("aggregate", END)
    
    return workflow.compile()
