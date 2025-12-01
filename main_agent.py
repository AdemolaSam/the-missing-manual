"""
Documently - AI Documentation Intelligence Agent
Main orchestrator agent that coordinates all sub-agents

This demonstrates:
- Multi-agent system (Sequential + Parallel agents)
- Custom tools + MCP integration
- Long-running operations with pause/resume
- Session & state management
- Memory bank for context persistence
- Observability (logging, tracing)
- Context engineering
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# ADK imports
from google.generativeai import configure
from google.generativeai.types import GenerationConfig
import google.generativeai as genai

# Setup logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class DocumentlyMemoryBank:
    """
    Long-term memory for storing user preferences and analysis history
    Demonstrates: Memory Bank pattern for context persistence
    """
    def __init__(self):
        self.user_preferences = {}
        self.analysis_history = []
        self.tool_insights = {}
        logger.info("Memory Bank initialized")
    
    def store_analysis(self, tool_name: str, results: Dict[str, Any]):
        """Store analysis results for future reference"""
        self.analysis_history.append({
            'tool': tool_name,
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        logger.info(f"Stored analysis for {tool_name}")
    
    def get_previous_analysis(self, tool_name: str) -> Optional[Dict]:
        """Retrieve previous analysis if available"""
        for analysis in reversed(self.analysis_history):
            if analysis['tool'].lower() == tool_name.lower():
                logger.info(f"Retrieved cached analysis for {tool_name}")
                return analysis['results']
        return None
    
    def update_tool_insights(self, tool_name: str, insights: Dict):
        """Accumulate insights about tools over time"""
        if tool_name not in self.tool_insights:
            self.tool_insights[tool_name] = []
        self.tool_insights[tool_name].append(insights)


class SessionManager:
    """
    Manages session state for long-running operations
    Demonstrates: Session & state management, pause/resume capability
    """
    def __init__(self):
        self.sessions = {}
        self.current_session_id = None
        logger.info("Session Manager initialized")
    
    def create_session(self, tool_name: str) -> str:
        """Create new analysis session"""
        session_id = f"session_{datetime.now().timestamp()}"
        self.sessions[session_id] = {
            'tool_name': tool_name,
            'status': 'active',
            'created_at': datetime.now().isoformat(),
            'steps_completed': [],
            'current_step': None,
            'results': {}
        }
        self.current_session_id = session_id
        logger.info(f"Created session {session_id} for {tool_name}")
        return session_id
    
    def update_session_step(self, session_id: str, step: str, result: Any):
        """Update session with completed step"""
        if session_id in self.sessions:
            self.sessions[session_id]['steps_completed'].append(step)
            self.sessions[session_id]['results'][step] = result
            logger.info(f"Session {session_id}: Completed step '{step}'")
    
    def pause_session(self, session_id: str):
        """Pause session for later resumption"""
        if session_id in self.sessions:
            self.sessions[session_id]['status'] = 'paused'
            logger.info(f"Session {session_id} paused")
    
    def resume_session(self, session_id: str):
        """Resume paused session"""
        if session_id in self.sessions:
            self.sessions[session_id]['status'] = 'active'
            logger.info(f"Session {session_id} resumed")
            return self.sessions[session_id]
    
    def get_session_state(self, session_id: str) -> Dict:
        """Get current session state"""
        return self.sessions.get(session_id, {})


class DocumentationSearchAgent:
    """
    Agent responsible for searching official documentation
    Demonstrates: Individual agent with specific responsibility
    """
    def __init__(self, api_key: str):
        configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("Documentation Search Agent initialized")
    
    async def search_official_docs(self, tool_name: str) -> Dict[str, Any]:
        """Search and analyze official documentation"""
        logger.info(f"Searching official docs for {tool_name}")
        
        # In production, this would use web search tool or MCP
        # For demo, using Gemini with structured prompt
        prompt = f"""
        Analyze the official documentation for {tool_name}.
        Provide:
        1. Purpose and overview (2-3 sentences)
        2. Key features (list of 4-5 main features)
        3. Installation command
        4. Basic usage example (code snippet)
        5. Complexity level (Beginner/Intermediate/Advanced)
        
        Return as JSON with keys: purpose, features, installation, usage, complexity
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=GenerationConfig(temperature=0.3)
            )
            
            # Parse response (simplified for demo)
            result = {
                'source': 'official_docs',
                'tool': tool_name,
                'summary': response.text,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Official docs search completed for {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error searching docs: {str(e)}")
            return {'error': str(e)}


class YouTubeAnalysisAgent:
    """
    Agent for analyzing YouTube tutorials and extracting insights
    Demonstrates: Specialized agent for video content analysis
    """
    def __init__(self, api_key: str):
        configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("YouTube Analysis Agent initialized")
    
    async def analyze_tutorials(self, tool_name: str) -> List[Dict[str, Any]]:
        """Find and analyze top YouTube tutorials"""
        logger.info(f"Analyzing YouTube tutorials for {tool_name}")
        
        # In production, this would:
        # 1. Use YouTube API to search for tutorials
        # 2. Transcribe top videos
        # 3. Extract code snippets and key timestamps
        # 4. Identify common patterns
        
        prompt = f"""
        Generate insights about top YouTube tutorials for {tool_name}.
        For 2-3 hypothetical top tutorials, provide:
        - Tutorial title
        - Channel name
        - Key takeaway
        - Important timestamp
        - Code snippet availability
        
        Return as JSON array.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=GenerationConfig(temperature=0.5)
            )
            
            result = {
                'source': 'youtube',
                'tool': tool_name,
                'tutorials': response.text,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"YouTube analysis completed for {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing YouTube: {str(e)}")
            return {'error': str(e)}


class GitHubIntelligenceAgent:
    """
    Agent for mining GitHub issues, discussions, and repositories
    Demonstrates: Real-world usage pattern extraction
    """
    def __init__(self, api_key: str):
        configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("GitHub Intelligence Agent initialized")
    
    async def gather_github_insights(self, tool_name: str) -> Dict[str, Any]:
        """Gather insights from GitHub issues and repositories"""
        logger.info(f"Gathering GitHub insights for {tool_name}")
        
        # In production, this would:
        # 1. Use GitHub API to fetch issues
        # 2. Analyze issue patterns and solutions
        # 3. Find popular repositories using the tool
        # 4. Extract common usage patterns
        
        prompt = f"""
        Generate GitHub intelligence for {tool_name}:
        1. Common issues (3-4 issues with solutions)
        2. Popular usage patterns (2-3 patterns with examples)
        3. Gotchas and pitfalls to avoid
        
        Return as JSON with keys: common_issues, usage_patterns, gotchas
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=GenerationConfig(temperature=0.4)
            )
            
            result = {
                'source': 'github',
                'tool': tool_name,
                'insights': response.text,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"GitHub intelligence gathered for {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error gathering GitHub insights: {str(e)}")
            return {'error': str(e)}


class RealWorldUsageAgent:
    """
    Agent for analyzing real-world production usage
    Demonstrates: Production pattern extraction from OSS projects
    """
    def __init__(self, api_key: str):
        configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("Real World Usage Agent initialized")
    
    async def analyze_production_usage(self, tool_name: str) -> List[Dict[str, Any]]:
        """Analyze how tool is used in production projects"""
        logger.info(f"Analyzing production usage for {tool_name}")
        
        prompt = f"""
        Identify 2-3 real-world projects using {tool_name} in production:
        For each project:
        - Project name
        - Use case
        - Implementation approach
        - Key learnings
        - Stars/popularity
        
        Return as JSON array.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=GenerationConfig(temperature=0.5)
            )
            
            result = {
                'source': 'real_world',
                'tool': tool_name,
                'projects': response.text,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Production usage analysis completed for {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing production usage: {str(e)}")
            return {'error': str(e)}


class SynthesisAgent:
    """
    Agent that synthesizes insights from all other agents
    Demonstrates: Multi-agent coordination and result synthesis
    """
    def __init__(self, api_key: str):
        configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("Synthesis Agent initialized")
    
    async def synthesize_results(
        self, 
        tool_name: str,
        doc_results: Dict,
        youtube_results: Dict,
        github_results: Dict,
        realworld_results: Dict
    ) -> Dict[str, Any]:
        """Synthesize all gathered intelligence into coherent documentation"""
        logger.info(f"Synthesizing results for {tool_name}")
        
        # Context engineering: Combine results efficiently
        context = f"""
        Tool: {tool_name}
        
        Official Docs Summary: {doc_results.get('summary', 'N/A')}
        YouTube Insights: {youtube_results.get('tutorials', 'N/A')}
        GitHub Intelligence: {github_results.get('insights', 'N/A')}
        Production Usage: {realworld_results.get('projects', 'N/A')}
        """
        
        prompt = f"""
        Based on the following intelligence gathered about {tool_name}:
        
        {context}
        
        Create a comprehensive developer onboarding guide that includes:
        1. Clear overview and purpose
        2. Quick start guide with code
        3. Key insights from videos
        4. Common pitfalls from GitHub issues
        5. Real-world usage patterns
        6. Best practices
        
        Structure this for maximum developer productivity.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=2048
                )
            )
            
            result = {
                'tool': tool_name,
                'comprehensive_guide': response.text,
                'sources_used': ['official_docs', 'youtube', 'github', 'real_world'],
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Synthesis completed for {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error during synthesis: {str(e)}")
            return {'error': str(e)}


class DocumentlyOrchestrator:
    """
    Main orchestrator agent that coordinates all sub-agents
    Demonstrates: Multi-agent orchestration, sequential and parallel execution
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Initialize sub-agents
        self.doc_agent = DocumentationSearchAgent(api_key)
        self.youtube_agent = YouTubeAnalysisAgent(api_key)
        self.github_agent = GitHubIntelligenceAgent(api_key)
        self.realworld_agent = RealWorldUsageAgent(api_key)
        self.synthesis_agent = SynthesisAgent(api_key)
        
        # Initialize memory and session management
        self.memory = DocumentlyMemoryBank()
        self.session_manager = SessionManager()
        
        logger.info("Documently Orchestrator initialized")
    
    async def analyze_tool(
        self, 
        tool_name: str,
        use_cache: bool = True,
        parallel_execution: bool = True
    ) -> Dict[str, Any]:
        """
        Main analysis workflow
        
        Args:
            tool_name: Name of tool/library to analyze
            use_cache: Whether to use cached results if available
            parallel_execution: Run agents in parallel vs sequential
        
        Returns:
            Comprehensive documentation intelligence
        """
        logger.info(f"Starting analysis for {tool_name}")
        
        # Check cache first (Memory Bank)
        if use_cache:
            cached = self.memory.get_previous_analysis(tool_name)
            if cached:
                logger.info(f"Returning cached results for {tool_name}")
                return cached
        
        # Create session for long-running operation
        session_id = self.session_manager.create_session(tool_name)
        
        try:
            if parallel_execution:
                # Parallel agent execution for faster results
                logger.info("Executing agents in parallel")
                
                doc_task = self.doc_agent.search_official_docs(tool_name)
                youtube_task = self.youtube_agent.analyze_tutorials(tool_name)
                github_task = self.github_agent.gather_github_insights(tool_name)
                realworld_task = self.realworld_agent.analyze_production_usage(tool_name)
                
                # Wait for all parallel tasks
                doc_results, youtube_results, github_results, realworld_results = \
                    await asyncio.gather(
                        doc_task,
                        youtube_task,
                        github_task,
                        realworld_task
                    )
            else:
                # Sequential execution
                logger.info("Executing agents sequentially")
                
                doc_results = await self.doc_agent.search_official_docs(tool_name)
                self.session_manager.update_session_step(session_id, 'docs', doc_results)
                
                youtube_results = await self.youtube_agent.analyze_tutorials(tool_name)
                self.session_manager.update_session_step(session_id, 'youtube', youtube_results)
                
                github_results = await self.github_agent.gather_github_insights(tool_name)
                self.session_manager.update_session_step(session_id, 'github', github_results)
                
                realworld_results = await self.realworld_agent.analyze_production_usage(tool_name)
                self.session_manager.update_session_step(session_id, 'realworld', realworld_results)
            
            # Synthesis phase - sequential (needs all previous results)
            logger.info("Starting synthesis phase")
            final_results = await self.synthesis_agent.synthesize_results(
                tool_name,
                doc_results,
                youtube_results,
                github_results,
                realworld_results
            )
            
            # Store in memory bank
            self.memory.store_analysis(tool_name, final_results)
            self.session_manager.update_session_step(session_id, 'synthesis', final_results)
            
            # Add raw insights for transparency
            final_results['raw_insights'] = {
                'official_docs': doc_results,
                'youtube': youtube_results,
                'github': github_results,
                'real_world': realworld_results
            }
            
            logger.info(f"Analysis completed for {tool_name}")
            return final_results
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            self.session_manager.pause_session(session_id)
            raise


# Example usage
async def main():
    """Example usage of Documently agent"""
    
    # Replace with your actual API key
    API_KEY = GEMINI_API_KEY
    
    # Initialize orchestrator
    documently = DocumentlyOrchestrator(API_KEY)
    
    # Analyze a tool
    tool_name = "Google ADK"
    
    print(f"\n🔍 Analyzing {tool_name}...")
    print("=" * 60)
    
    results = await documently.analyze_tool(
        tool_name=tool_name,
        use_cache=False,
        parallel_execution=True
    )
    
    print("\n✅ Analysis Complete!")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())