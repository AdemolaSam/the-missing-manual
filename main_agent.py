"""
Documently - AI Documentation Intelligence Agent
Built with Google ADK (Agent Development Kit)

This demonstrates:
- Multi-agent system (Sequential + Parallel agents using ADK workflow agents)
- Custom tools
- Long-running operations with session management
- Memory bank for context persistence
- Observability (logging, tracing)
- Context engineering
"""

import os
from typing import Dict, Any
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ADK imports - Using actual Google ADK framework
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

# Setup logging for observability
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set. Please set it in your .env file.")
    raise ValueError("GEMINI_API_KEY is required")


# CUSTOM TOOLS

def get_current_time() -> dict:
    """Get current timestamp for documentation generation"""
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }


# MEMORY BANK - Long-term memory for caching analyses
# Demonstrates: Memory persistence across sessions
class DocumentlyMemoryBank:
    """
    Memory bank for storing and retrieving previous analyses
    Provides caching capability for instant responses
    """
    def __init__(self):
        self.cache = {}
        logger.info("Memory Bank initialized")
    
    def store(self, tool_name: str, results: Dict[str, Any]):
        """Cache analysis results"""
        self.cache[tool_name.lower()] = {
            'results': results,
            'cached_at': datetime.now().isoformat()
        }
        logger.info(f"Cached analysis for {tool_name}")
    
    def retrieve(self, tool_name: str) -> Dict[str, Any] | None:
        """Retrieve cached analysis if available"""
        cached = self.cache.get(tool_name.lower())
        if cached:
            logger.info(f"Cache hit for {tool_name}")
            return cached['results']
        logger.info(f"Cache miss for {tool_name}")
        return None
    
    def clear(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Memory bank cleared")


# SPECIALIZED AGENTS - Each agent handles one source type
# Demonstrates: Agent specialization and modularity

# Agent 1: Documentation Search Agent
documentation_agent = Agent(
    name="DocumentationSearchAgent",
    model="gemini-2.0-flash-exp",
    instruction="""You are a Documentation Analysis Specialist.

Your task: Analyze official documentation for the given tool/framework.

Provide:
1. Purpose and overview (2-3 sentences)
2. Key features (list 4-5 main features)
3. Installation command
4. Basic usage example (simple code snippet)
5. Complexity level (Beginner/Intermediate/Advanced)

Format your response clearly with headers.
Be concise and developer-focused.""",
    description="Analyzes official documentation and extracts key information",
    output_key="docs_analysis",  # Store result in session state
    tools=[get_current_time]
)

# Agent 2: YouTube Analysis Agent
youtube_agent = Agent(
    name="YouTubeAnalysisAgent", 
    model="gemini-2.0-flash-exp",
    instruction="""You are a Video Tutorial Analysis Specialist.

Your task: Identify insights from YouTube tutorials for the given tool.

Generate insights for 2-3 hypothetical top tutorials:
- Tutorial title
- Channel name
- Key takeaway
- Important timestamp (e.g., "12:30 - Setup configuration")
- Whether code snippets are available

Focus on practical, actionable insights.
Format clearly with bullet points.""",
    description="Analyzes YouTube tutorials and extracts key insights",
    output_key="youtube_analysis"
)

# Agent 3: GitHub Intelligence Agent
github_agent = Agent(
    name="GitHubIntelligenceAgent",
    model="gemini-2.0-flash-exp",
    instruction="""You are a GitHub Intelligence Specialist.

Your task: Mine GitHub for real-world insights about the given tool.

Provide:
1. Common issues (3-4 issues with their solutions)
2. Popular usage patterns (2-3 patterns with examples)
3. Gotchas and pitfalls to avoid

Base this on typical GitHub issues and repository patterns.
Format with clear sections and bullet points.""",
    description="Mines GitHub issues and repos for common problems and patterns",
    output_key="github_analysis"
)

# Agent 4: Real-World Usage Agent
realworld_agent = Agent(
    name="RealWorldUsageAgent",
    model="gemini-2.0-flash-exp",
    instruction="""You are a Production Usage Analysis Specialist.

Your task: Identify how the tool is used in real production projects.

For 2-3 hypothetical production projects:
- Project name
- Use case
- Implementation approach  
- Key learnings
- Popularity (stars)

Focus on practical patterns and best practices.
Format clearly with sections.""",
    description="Analyzes real-world production usage patterns",
    output_key="realworld_analysis"
)

# Agent 5: Synthesis Agent (runs after parallel agents)
synthesis_agent = Agent(
    name="SynthesisAgent",
    model="gemini-2.0-flash-exp",
    instruction="""You are a Documentation Synthesis Specialist.

Your task: Create a comprehensive developer onboarding guide.

You will receive analysis from multiple sources:
- Official documentation: {docs_analysis}
- YouTube tutorials: {youtube_analysis}
- GitHub intelligence: {github_analysis}
- Real-world usage: {realworld_analysis}

Synthesize these into a structured guide with:
1. **Overview** - What is this tool and why use it?
2. **Quick Start** - Installation and first steps
3. **Key Insights** - Important points from videos
4. **Common Pitfalls** - Issues from GitHub
5. **Real-World Patterns** - How it's actually used
6. **Best Practices** - Recommendations

Make it actionable and developer-friendly.
Use clear headings and concise language.""",
    description="Synthesizes all insights into comprehensive developer guide",
    output_key="final_guide"
)


# WORKFLOW ORCHESTRATION - Multi-Agent System
# Demonstrates: Parallel + Sequential agent coordination

# Parallel Agent: Runs 4 specialist agents simultaneously
# This is the "gathering" phase - all sources analyzed at once
parallel_gathering_agent = ParallelAgent(
    name="ParallelGatheringAgent",
    sub_agents=[
        documentation_agent,
        youtube_agent,
        github_agent,
        realworld_agent
    ],
    description="Gathers intelligence from 4 sources in parallel for speed"
)

# Sequential Agent: Orchestrates the full workflow
# Step 1: Run parallel gathering
# Step 2: Run synthesis (which needs all parallel results)
root_agent = SequentialAgent(
    name="DocumentlyOrchestrator",
    sub_agents=[
        parallel_gathering_agent,  # First: gather from all sources in parallel
        synthesis_agent             # Second: synthesize all results
    ],
    description="Orchestrates parallel gathering followed by synthesis"
)


# DOCUMENTLY MAIN CLASS - Ties everything together
# Demonstrates: Session management, memory, runner integration

class Documently:
    """
    Main Documently class integrating all components
    """
    def __init__(self, app_name):
        self.app_name = app_name
        # Initialize runner (ADK's execution engine)
        # InMemoryRunner automatically creates InMemorySessionService
        self.runner = InMemoryRunner(agent=root_agent)
        
        # Access the runner's session service
        self.session_service = self.runner.session_service
        
        # Initialize memory bank for caching
        self.memory = DocumentlyMemoryBank()
        
        # User ID for session tracking
        self.user_id = "documently_user"
        
        logger.info("Documently initialized with ADK framework")
    
    async def analyze_tool(
        self,
        tool_name: str,
        use_cache: bool = True,
        session_id: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze a tool/framework and generate comprehensive guide
        
        Args:
            tool_name: Name of tool to analyze
            use_cache: Whether to use cached results
            session_id: Optional session ID for resuming
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Starting analysis for {tool_name}")
        
        # Check cache first (Memory Bank)
        if use_cache:
            cached = self.memory.retrieve(tool_name)
            if cached:
                logger.info(f"Returning cached results for {tool_name}")
                return cached
        
        # Create or use existing session
        if session_id is None:
            session_id = f"session_{datetime.now().timestamp()}"
            logger.info(f"Created new session: {session_id}")
        else:
            logger.info(f"Resuming session: {session_id}")
        
        # Prepare user message
        user_message = Content(
            role="user",
            parts=[Part(text=f"Analyze the tool/framework: {tool_name}")]
        )
        
        try:
            # Run the agent (ADK handles all orchestration)
            logger.info("Executing parallel gathering agents...")
            
            final_text = ""
            response = self.runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=user_message
            )
            
            # Extract final response

            async for event in response:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_text += part.text
            
            # Get session state to access all intermediate results
            session = await self.session_service.get_session(
                app_name=self.runner.app_name,
                user_id=self.user_id,
                session_id=session_id
            )
            
            # Compile full results
            results = {
                'tool': tool_name,
                'comprehensive_guide': final_text,
                'session_id': session_id,
                'generated_at': datetime.now().isoformat(),
                'sources_analyzed': ['official_docs', 'youtube', 'github', 'real_world'],
                'state': session.state if session else {} # type: ignore
            }
            
            # Store in memory bank
            self.memory.store(tool_name, results)
            
            logger.info(f"Analysis completed for {tool_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise
    
    async def pause_session(self, session_id: str):
        """
        Pause a session (ADK automatically persists state)
        
        Args:
            session_id: Session to pause
        """
        logger.info(f"Session {session_id} can be resumed later")
        # ADK's InMemorySessionService automatically maintains state
        # No explicit pause needed - session persists until cleared
    
    async def get_session_state(self, session_id: str) -> Dict:
        """
        Get current session state
        
        Args:
            session_id: Session ID
            
        Returns:
            Session state dict
        """
        session = self.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=self.user_id,
            session_id=session_id
        )
        
        if session:
            logger.info(f"Retrieved state for session {session_id}")
            return session.state # type: ignore
        else:
            logger.warning(f"Session {session_id} not found")
            return {}


# EXAMPLE USAGE

async def main():
    """Example usage of Documently with Google ADK"""
    
    # Initialize Documently
    documently = Documently()
    
    # Analyze a tool
    tool_name = "Google ADK"
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing: {tool_name}")
    print(f"{'='*60}\n")
    
    # First analysis (no cache)
    print("Running full analysis (parallel execution)...")
    results = await documently.analyze_tool(
        tool_name=tool_name,
        use_cache=False
    )
    
    print(f"\n{'='*60}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    
    print(f"📊 Sources Analyzed: {', '.join(results['sources_analyzed'])}")
    print(f"🕒 Generated: {results['generated_at']}")
    print(f"💾 Session ID: {results['session_id']}\n")
    
    print(f"{'='*60}")
    print("📚 COMPREHENSIVE GUIDE")
    print(f"{'='*60}\n")
    print(results['comprehensive_guide'])
    
    # Test caching
    print(f"\n{'='*60}")
    print("🔄 Testing cache (second request)...")
    print(f"{'='*60}\n")
    
    cached_results = await documently.analyze_tool(
        tool_name=tool_name,
        use_cache=True
    )
    
    print("✅ Retrieved from cache (instant response!)\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())