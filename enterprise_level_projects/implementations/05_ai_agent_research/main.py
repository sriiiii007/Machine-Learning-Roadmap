"""
Autonomous Research Agent
AI agent that researches topics and generates comprehensive reports.
"""

# Note: Install required packages:
# pip install langchain openai duckduckgo-search

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("AUTONOMOUS RESEARCH AGENT")
print("=" * 70)

# ============================================================================
# AGENT SETUP
# ============================================================================
print("\n" + "=" * 70)
print("Step 1: Agent Setup")
print("=" * 70)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  OpenAI API key not found.")
    print("This agent requires OpenAI API for full functionality.")
    print("Set OPENAI_API_KEY environment variable to use this agent.")
    print("\nDemonstrating agent structure without API calls...")
    use_agent = False
else:
    use_agent = True
    llm = OpenAI(temperature=0.7)
    print("✅ OpenAI API configured")

# ============================================================================
# TOOLS SETUP
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Tools Setup")
print("=" * 70)

# Search tool
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching the internet for current information, facts, or recent events. Input should be a search query."
    )
]

print(f"Tools configured: {len(tools)}")
print("  - Search: Internet search capability")

# ============================================================================
# AGENT INITIALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Agent Initialization")
print("=" * 70)

if use_agent:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    print("✅ Research agent initialized")
else:
    agent = None
    print("⚠️  Agent structure ready (requires API key)")

# ============================================================================
# RESEARCH FUNCTION
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: Research Function")
print("=" * 70)

def research_topic(topic, agent=None):
    """Research a topic and generate a report"""
    
    if not agent:
        return {
            "topic": topic,
            "status": "requires_openai_api",
            "message": "Set OPENAI_API_KEY to use research agent"
        }
    
    # Research questions
    research_questions = [
        f"What is {topic}?",
        f"What are the key aspects of {topic}?",
        f"What are recent developments in {topic}?",
        f"What are the main challenges in {topic}?"
    ]
    
    print(f"\nResearching: {topic}")
    print(f"Research questions: {len(research_questions)}")
    
    # Conduct research
    research_results = []
    for question in research_questions:
        print(f"\n  Researching: {question}")
        try:
            result = agent.run(question)
            research_results.append({
                "question": question,
                "answer": result
            })
        except Exception as e:
            print(f"  Error: {e}")
            research_results.append({
                "question": question,
                "answer": f"Error researching: {str(e)}"
            })
    
    return {
        "topic": topic,
        "research_results": research_results,
        "status": "completed"
    }

# ============================================================================
# REPORT GENERATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 5: Report Generation")
print("=" * 70)

def generate_report(research_data, llm=None):
    """Generate comprehensive report from research"""
    
    if not llm:
        return "Report generation requires LLM"
    
    prompt = PromptTemplate(
        input_variables=["topic", "research"],
        template="""
        Based on the following research, create a comprehensive report on {topic}.
        
        Research Findings:
        {research}
        
        Create a well-structured report with:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis
        4. Conclusions
        5. Recommendations
        
        Report:
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    research_text = "\n\n".join([
        f"Q: {r['question']}\nA: {r['answer']}"
        for r in research_data['research_results']
    ])
    
    report = chain.run(
        topic=research_data['topic'],
        research=research_text
    )
    
    return report

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
print("\n" + "=" * 70)
print("Step 6: Example Usage")
print("=" * 70)

if use_agent:
    # Research a topic
    topic = "artificial intelligence trends 2024"
    print(f"\nExample: Researching '{topic}'")
    
    research_data = research_topic(topic, agent)
    
    if research_data['status'] == 'completed':
        print("\n" + "="*70)
        print("RESEARCH RESULTS")
        print("="*70)
        
        for i, result in enumerate(research_data['research_results'], 1):
            print(f"\n{i}. {result['question']}")
            print(f"   {result['answer'][:200]}...")
        
        # Generate report
        print("\n" + "="*70)
        print("GENERATING REPORT...")
        print("="*70)
        
        report = generate_report(research_data, llm)
        print(f"\n{report[:500]}...")
        
        # Save report
        with open(f"research_report_{topic.replace(' ', '_')}.txt", "w") as f:
            f.write(f"Research Report: {topic}\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        print(f"\n✅ Report saved to: research_report_{topic.replace(' ', '_')}.txt")
else:
    print("\n" + "="*70)
    print("AGENT STRUCTURE DEMONSTRATION")
    print("="*70)
    
    print("""
    This agent would:
    
    1. RECEIVE RESEARCH TOPIC
       Example: "artificial intelligence trends 2024"
    
    2. GENERATE RESEARCH QUESTIONS
       - What is AI?
       - What are current AI trends?
       - What are recent developments?
    
    3. SEARCH FOR INFORMATION
       - Use search tool to find information
       - Gather multiple sources
       - Extract key facts
    
    4. SYNTHESIZE FINDINGS
       - Combine information from multiple sources
       - Identify key themes
       - Organize information
    
    5. GENERATE REPORT
       - Create comprehensive report
       - Include executive summary
       - Provide recommendations
    
    To use this agent:
    1. Set OPENAI_API_KEY environment variable
    2. Run: python main.py
    3. Agent will research and generate report
    """)

# ============================================================================
# AGENT CAPABILITIES
# ============================================================================
print("\n" + "=" * 70)
print("Step 7: Agent Capabilities")
print("=" * 70)

print("""
This Research Agent can:

✅ AUTONOMOUS RESEARCH
   - Generates research questions
   - Searches multiple sources
   - Gathers comprehensive information

✅ INFORMATION SYNTHESIS
   - Combines information from multiple sources
   - Identifies key themes
   - Organizes findings

✅ REPORT GENERATION
   - Creates structured reports
   - Includes analysis
   - Provides recommendations

✅ EXTENSIBILITY
   - Add more tools (databases, APIs)
   - Customize research questions
   - Adjust report format

FUTURE ENHANCEMENTS:
- Multi-step research (iterative)
- Source citation
- Fact-checking
- Multi-language support
- Specialized domain research
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
✅ Research Agent Structure Created!
{'✅ Agent ready to use' if use_agent else '⚠️  Set OPENAI_API_KEY to activate agent'}
✅ Tools configured: Search
✅ Report generation capability

Next Steps:
1. Set OPENAI_API_KEY for full functionality
2. Add more research tools (databases, APIs)
3. Implement multi-step research
4. Add source citation
5. Create web interface
6. Deploy as service
7. Add monitoring and logging
""")

print("=" * 70)

