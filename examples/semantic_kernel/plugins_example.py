"""
Semantic Kernel plugins example.

Demonstrates creating and using plugins (collections of functions).
"""

import os
import asyncio
from llm_interface.src.manager import LLMManager
from llm_interface.src.core.config import LLMConfig, ProviderType
from llm_interface.src.framework.semantic_kernel import SemanticKernelAdapter


# Define native functions for weather plugin
def get_current_weather(city: str) -> str:
    """Get current weather for a city (mock implementation)."""
    # In real implementation, this would call a weather API
    weather_data = {
        "Seattle": "Rainy, 55°F",
        "San Francisco": "Sunny, 65°F",
        "New York": "Cloudy, 45°F",
        "London": "Foggy, 50°F",
        "Tokyo": "Clear, 60°F"
    }
    return weather_data.get(city, f"Weather data not available for {city}")


def get_temperature(city: str) -> str:
    """Get temperature for a city (mock implementation)."""
    temps = {
        "Seattle": "55°F (13°C)",
        "San Francisco": "65°F (18°C)",
        "New York": "45°F (7°C)",
        "London": "50°F (10°C)",
        "Tokyo": "60°F (16°C)"
    }
    return temps.get(city, "Temperature data not available")


# Define native functions for math plugin
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


async def main():
    """Run plugin examples."""
    
    # Initialize LLM Manager with provider
    manager = LLMManager()
    
    # Configure OpenAI provider
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=500
    )
    manager.add_provider("gpt4", openai_config)
    
    # Create Semantic Kernel adapter
    adapter = SemanticKernelAdapter(manager)
    
    print("=" * 80)
    print("Semantic Kernel Plugins Example")
    print("=" * 80)
    print(f"Using provider: {openai_config.model_name}")
    print()
    
    # Create kernel
    kernel = adapter.create_kernel("gpt4", kernel_id="main")
    print(f"✅ Created kernel: main")
    
    # Example 1: Weather Plugin with Native Functions
    print("\n" + "=" * 80)
    print("Example 1: Weather Plugin (Native Functions)")
    print("=" * 80)
    
    weather_plugin = adapter.create_plugin(
        kernel=kernel,
        plugin_name="weather",
        functions={
            "get_current": get_current_weather,
            "get_temperature": get_temperature
        }
    )
    
    print("✅ Created weather plugin with 2 functions")
    
    # Use weather plugin functions
    result = await adapter.invoke_function(
        kernel=kernel,
        function="weather.get_current",
        arguments={"city": "Seattle"}
    )
    print(f"\nCurrent weather in Seattle: {result}")
    
    result = await adapter.invoke_function(
        kernel=kernel,
        function="weather.get_temperature",
        arguments={"city": "Tokyo"}
    )
    print(f"Temperature in Tokyo: {result}")
    
    # Example 2: Math Plugin with Native Functions
    print("\n" + "=" * 80)
    print("Example 2: Math Plugin (Native Functions)")
    print("=" * 80)
    
    math_plugin = adapter.create_plugin(
        kernel=kernel,
        plugin_name="math",
        functions={
            "add": add_numbers,
            "multiply": multiply_numbers
        }
    )
    
    print("✅ Created math plugin with 2 functions")
    
    result = await adapter.invoke_function(
        kernel=kernel,
        function="math.add",
        arguments={"a": 15, "b": 27}
    )
    print(f"\n15 + 27 = {result}")
    
    result = await adapter.invoke_function(
        kernel=kernel,
        function="math.multiply",
        arguments={"a": 8, "b": 9}
    )
    print(f"8 × 9 = {result}")
    
    # Example 3: Content Plugin with Semantic Functions
    print("\n" + "=" * 80)
    print("Example 3: Content Plugin (Semantic Functions)")
    print("=" * 80)
    
    # Add semantic functions to a plugin
    adapter.create_semantic_function(
        kernel=kernel,
        prompt="Generate a creative title for an article about {{$topic}}",
        function_name="generate_title",
        plugin_name="content",
        description="Generates article titles",
        max_tokens=50
    )
    
    adapter.create_semantic_function(
        kernel=kernel,
        prompt="Write a brief introduction paragraph about {{$topic}}",
        function_name="generate_intro",
        plugin_name="content",
        description="Generates introductions",
        max_tokens=150
    )
    
    print("✅ Created content plugin with 2 semantic functions")
    
    topic = "sustainable energy"
    
    title = await adapter.invoke_function(
        kernel=kernel,
        function="content.generate_title",
        arguments={"topic": topic}
    )
    print(f"\nGenerated Title: {title}")
    
    intro = await adapter.invoke_function(
        kernel=kernel,
        function="content.generate_intro",
        arguments={"topic": topic}
    )
    print(f"\nGenerated Intro: {intro}")
    
    # Example 4: Mixed Plugin (Semantic + Native)
    print("\n" + "=" * 80)
    print("Example 4: Mixed Plugin (Semantic + Native Functions)")
    print("=" * 80)
    
    # Add native function
    def word_count(text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    adapter.create_native_function(
        kernel=kernel,
        function=word_count,
        function_name="count_words",
        plugin_name="text_analysis",
        description="Counts words in text"
    )
    
    # Add semantic function to same plugin
    adapter.create_semantic_function(
        kernel=kernel,
        prompt="Extract the main keywords from this text:\n\n{{$text}}",
        function_name="extract_keywords",
        plugin_name="text_analysis",
        description="Extracts keywords from text",
        max_tokens=100
    )
    
    print("✅ Created text_analysis plugin with native + semantic functions")
    
    sample_text = "Artificial intelligence and machine learning are transforming how we build software applications."
    
    count = await adapter.invoke_function(
        kernel=kernel,
        function="text_analysis.count_words",
        arguments={"text": sample_text}
    )
    print(f"\nSample text: '{sample_text}'")
    print(f"Word count: {count}")
    
    keywords = await adapter.invoke_function(
        kernel=kernel,
        function="text_analysis.extract_keywords",
        arguments={"text": sample_text}
    )
    print(f"Keywords: {keywords}")
    
    # List all plugins and functions
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print(f"\nCreated Plugins: {adapter.list_plugins()}")
    print(f"\nAll Functions:")
    for func_name in adapter.list_functions():
        print(f"  - {func_name}")
    
    print()
    print("=" * 80)
    print("Plugins Example Complete!")
    print("=" * 80)


async def example_plugin_workflow():
    """Example showing a workflow using multiple plugins."""
    
    manager = LLMManager()
    
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7
    )
    manager.add_provider("gpt4", openai_config)
    
    adapter = SemanticKernelAdapter(manager)
    kernel = adapter.create_kernel("gpt4")
    
    print("=" * 80)
    print("Plugin Workflow Example")
    print("=" * 80)
    
    # Create a research assistant workflow
    # Plugin 1: Research
    adapter.create_semantic_function(
        kernel=kernel,
        prompt="Provide 3 key facts about {{$topic}}",
        function_name="get_facts",
        plugin_name="research",
        max_tokens=200
    )
    
    # Plugin 2: Writing
    adapter.create_semantic_function(
        kernel=kernel,
        prompt="Write a summary paragraph using these facts:\n\n{{$facts}}",
        function_name="write_summary",
        plugin_name="writing",
        max_tokens=250
    )
    
    # Plugin 3: Formatting
    def add_markdown_heading(text: str, heading: str) -> str:
        """Add markdown heading to text."""
        return f"# {heading}\n\n{text}"
    
    adapter.create_native_function(
        kernel=kernel,
        function=add_markdown_heading,
        function_name="add_heading",
        plugin_name="formatting"
    )
    
    print("✅ Created 3-plugin workflow: research → writing → formatting")
    
    # Execute workflow
    topic = "quantum computing"
    print(f"\nGenerating article about: {topic}")
    print("-" * 40)
    
    # Step 1: Research
    facts = await adapter.invoke_function(
        kernel=kernel,
        function="research.get_facts",
        arguments={"topic": topic}
    )
    print(f"\n1. Facts gathered:\n{facts}")
    
    # Step 2: Writing
    summary = await adapter.invoke_function(
        kernel=kernel,
        function="writing.write_summary",
        arguments={"facts": str(facts)}
    )
    print(f"\n2. Summary written:\n{summary}")
    
    # Step 3: Formatting
    article = await adapter.invoke_function(
        kernel=kernel,
        function="formatting.add_heading",
        arguments={"text": str(summary), "heading": topic.title()}
    )
    print(f"\n3. Final article:\n{article}")
    
    print()
    print("=" * 80)
    print("Workflow Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Run main example
    asyncio.run(main())
    
    print("\n\n")
    
    # Run workflow example
    asyncio.run(example_plugin_workflow())
