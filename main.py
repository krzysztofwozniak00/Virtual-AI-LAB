import io
import json
import os
import sys
import asyncio
from turtle import goto
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Annotated, Literal
from pydantic import BaseModel, Field
import requests
import arxiv
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.types import Command
from loguru import logger
from PIL import Image
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
import openai
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus.flowables import HRFlowable
import base64

# Load environment variables
# Load environment variables
load_dotenv()

# Helper function to safely parse environment variables with comment handling
def safe_parse_env_var(key, default_value, var_type=str):
    """Safely parse environment variable, removing comments and handling errors."""
    try:
        raw_value = os.getenv(key, str(default_value))
        # Remove inline comments (everything after #)
        clean_value = raw_value.split('#')[0].strip()
        
        if var_type == int:
            return int(clean_value)
        elif var_type == bool:
            return clean_value.lower() in ('true', '1', 'yes', 'on')
        else:
            return clean_value
    except (ValueError, TypeError, AttributeError):
        print(f"Warning: Could not parse {key}={raw_value}, using default: {default_value}")
        return default_value

# Set environment variables safely
os.environ["TAVILY_API_KEY"] = safe_parse_env_var("TAVILY_API_KEY", "")
os.environ["OPENAI_API_KEY"] = safe_parse_env_var("OPENAI_API_KEY", "")  


# Parse VERBOSE safely
VERBOSE = safe_parse_env_var("VERBOSE", 1, int) == 1

console = Console()

#  State definition for R&D Pipeline
class RnDState(MessagesState):
    """ state for R&D multi-agent system with comprehensive tracking."""
    request: str
    workflow_type: str  # "new_product" or "product_improvement" 
    research_brief_path: Optional[str] = None
    idea_list_path: Optional[str] = None
    selected_idea_path: Optional[str] = None
    design_spec_path: Optional[str] = None
    concept_image_path: Optional[str] = None
    test_plan_path: Optional[str] = None
    final_report_path: Optional[str] = None
    current_agent: str = "supervisor"
    iterations: int = 0

def get_llm_config():
    """Get LLM configuration from environment variables"""
    model = os.getenv("EXPERIMENT_LLM_MODEL", "gpt-4o")
    temperature = float(os.getenv("EXPERIMENT_TEMPERATURE", "0.7"))
    return model, temperature

#  Tools with Increased Capabilities
@tool
@traceable
def get_current_datetime() -> str:
    """Retrieve the current date and time in ISO format for documentation purposes."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"📅 Current timestamp: {current_time}")
    return current_time


@tool
@traceable
def tavily_search(query: str = Field(description="The query to search")) -> str:
    """Conduct comprehensive Web search using Tavily search engine."""
    logger.info(f"🔍 Web Search: {query}")
    try:
        search = TavilySearchResults(max_results=10)
        results = search.invoke(query)
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Search error: {str(e)}"

@tool
@traceable
def market_intelligence_search(query: str = Field(description="The query to search")) -> str:
    """Conduct comprehensive market intelligence search using Tavily search engine."""
    logger.info(f"🔍 Market Intelligence Search: {query}")
    try:
        search = TavilySearchResults(max_results=10)
        results = search.invoke(query)
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"Market Intelligence search error: {e}")
        return f"Search error: {str(e)}"


@tool
@traceable
def arxiv_research_search(query: str = Field(description="ArXiv research query")) -> str:
    """Search ArXiv preprint repository for latest scientific research papers."""
    logger.info(f"🔬 ArXiv Research Search: {query}")
    try:
        Client = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in Client.results():
            results.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "abstract": result.summary[:800] + "..." if len(result.summary) > 800 else result.summary,
                "categories": result.categories,
                "url": result.entry_id,
                "doi": result.doi
            })
        
        return json.dumps({"papers": results}, indent=2)
    except Exception as e:
        logger.error(f"ArXiv search error: {e}")
        return f"ArXiv search error: {str(e)}"


@tool
@traceable
def save_research_brief(content: str = Field(description="Research brief markdown content")) -> str:
    """Save the comprehensive market and technical research brief as a markdown file."""
    logger.info("📄 Saving research brief")
    Path("output").mkdir(exist_ok=True)
    
    with open("output/01_research_brief.md", "w", encoding="utf-8") as f:
        f.write(content)
    
    return "output/01_research_brief.md"


@tool
@traceable
def save_idea_concepts(content: str = Field(description="Idea concepts markdown content")) -> str:
    """Save the generated product concepts as a markdown file."""
    logger.info("💡 Saving idea concepts")
    Path("output").mkdir(exist_ok=True)
    
    with open("output/02_idea_concepts.md", "w", encoding="utf-8") as f:
        f.write(content)
    
    return "output/02_idea_concepts.md"


@tool
@traceable
def save_selected_concept(content: str = Field(description="Selected concept markdown content")) -> str:
    """Save the selected concept with detailed analysis as a markdown file."""
    logger.info("✅ Saving selected concept")
    Path("output").mkdir(exist_ok=True)
    
    with open("output/03_selected_concept.md", "w", encoding="utf-8") as f:
        f.write(content)
    
    return "output/03_selected_concept.md"


@tool
@traceable
def save_technical_specification(content: str = Field(description="Technical specification markdown content")) -> str:
    """Save the detailed technical specifications as a markdown file."""
    logger.info("📐 Saving technical specifications")
    Path("output").mkdir(exist_ok=True)
    
    with open("output/04_technical_specifications.md", "w", encoding="utf-8") as f:
        f.write(content)
    
    return "output/04_technical_specifications.md"


@tool
@traceable
def save_testing_plan(content: str = Field(description="Testing plan markdown content")) -> str:
    """Save the comprehensive testing and validation plan as a markdown file."""
    logger.info("🧪 Saving testing plan")
    Path("output").mkdir(exist_ok=True)
    
    with open("output/05_testing_plan.md", "w", encoding="utf-8") as f:
        f.write(content)
    
    return "output/05_testing_plan.md"


@tool
@traceable
def generate_concept_visualization(prompt: str = Field(description="Detailed image generation prompt")) -> str:
    """Generate a professional concept visualization using DALL-E 3."""
    logger.info("🎨 Generating concept visualization")
    try:
        client = openai.OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            n=1,
        )
        
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        
        Path("output").mkdir(exist_ok=True)
        image_path = "output/06_concept_visualization.png"
        
        with open(image_path, "wb") as f:
            f.write(image_response.content)
        
        return image_path
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return f"Image generation error: {str(e)}"



@tool
@traceable
def human_concept_selection(
    state: Annotated[dict, InjectedState],
    concepts_markdown: str = Field(description="Markdown formatted concepts for selection")
) -> str:
    """Present concept options to human for professional selection and feedback."""
    console = Console()
    
    # Display concepts for selection
    console.print("\n[bold cyan]CONCEPT SELECTION REQUIRED[/bold cyan]")
    console.print("[dim]Please review the following concept options:[/dim]\n")
    
    # Render markdown content
    md = Markdown(concepts_markdown)
    console.print(md)
    
    # Parse concepts for selection (assuming numbered format)
    lines = concepts_markdown.split('\n')
    concept_count = len([line for line in lines if line.startswith('## ')])
    
    console.print(f"\n[bold]Available concepts: 1-{concept_count}[/bold]")
    
    while True:
        choice = Prompt.ask(
            "Enter the number of your preferred concept", 
            choices=[str(i) for i in range(1, concept_count + 1)]
        )
        
        console.print(f"[green]✓ Concept {choice} selected[/green]")
        return f"Selected concept: {choice}"


@tool 
@traceable
def compile_comprehensive_report() -> str:
    """Compile all research outputs into an executive-quality comprehensive PDF report"""
    logger.info("📑 Compiling comprehensive R&D report with  formatting")
    
    # Read all markdown files and prepare content
    output_files = [
        ("output/01_research_brief.md", "Market Research & Technical Analysis"),
        ("output/02_idea_concepts.md", "Innovation Concepts & Development"), 
        ("output/03_selected_concept.md", "Selected Concept Analysis"),
        ("output/04_technical_specifications.md", "Technical Specifications & Architecture"),
        ("output/05_testing_plan.md", "Testing & Validation Framework")
    ]
    
    Path("output").mkdir(exist_ok=True)
    
    # Create document with  formatting
    doc = SimpleDocTemplate(
        "output/07_comprehensive_report.pdf", 
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=0.75*inch,
        title="Comprehensive R&D Analysis Report",
        author="R&D Multi-Agent System"
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    #  Custom Styles
    title_style = ParagraphStyle(
        'ExecutiveTitle',
        parent=styles['Title'],
        fontSize=32,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=0.5*inch,
        spaceBefore=0.2*inch,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'ExecutiveSubtitle',
        parent=styles['Normal'],
        fontSize=18,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=0.3*inch,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading1_style = ParagraphStyle(
        'ExecutiveHeading1',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=0.25*inch,
        spaceBefore=0.4*inch,
        borderWidth=2,
        borderColor=colors.HexColor('#3182ce'),
        borderPadding=10,
        backColor=colors.HexColor('#ebf8ff'),
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'ExecutiveHeading2',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=0.15*inch,
        spaceBefore=0.25*inch,
        leftIndent=0.1*inch,
        borderWidth=1,
        borderColor=colors.HexColor('#a0aec0'),
        borderPadding=5,
        fontName='Helvetica-Bold'
    )
    
    heading3_style = ParagraphStyle(
        'ExecutiveHeading3',
        parent=styles['Heading3'],
        fontSize=16,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=0.1*inch,
        spaceBefore=0.2*inch,
        leftIndent=0.2*inch,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'ExecutiveBody',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_JUSTIFY,
        spaceAfter=0.12*inch,
        spaceBefore=0.02*inch,
        leftIndent=0.1*inch,
        rightIndent=0.1*inch,
        fontName='Helvetica'
    )
    
    emphasis_style = ParagraphStyle(
        'ExecutiveEmphasis',
        parent=body_style,
        backColor=colors.HexColor('#f7fafc'),
        borderWidth=1,
        borderColor=colors.HexColor('#e2e8f0'),
        borderPadding=8,
        leftIndent=0.2*inch,
        rightIndent=0.2*inch
    )
    
    # Executive Cover Page
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("COMPREHENSIVE R&D ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Add horizontal line
    story.append(HRFlowable(width="100%", thickness=3, color=colors.HexColor('#3182ce')))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Professional Multi-Agent Research & Development Analysis", subtitle_style))
    story.append(Spacer(1, 1*inch))
    
    # Executive Summary Box
    exec_summary = f"""
    <b>Executive Summary:</b><br/>
    This comprehensive report presents the findings from a multi-agent R&D analysis system, 
    incorporating market intelligence, technical research, innovation development, and strategic planning. 
    The analysis leverages advanced AI agents specialized in market research, academic literature review, 
    concept development, technical specification, and validation planning to deliver actionable insights 
    for strategic decision-making.
    """
    story.append(Paragraph(exec_summary, emphasis_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Report Details
    details_data = [
        ['Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
        ['Analysis Scope:', 'Comprehensive Multi-Agent R&D Pipeline'],
        ['Research Sources:', 'Web Search, Market Intelligence, Academic Literature, ArXiv Repository'],
        ['Methodology:', 'AI-Driven Multi-Agent Collaborative Analysis'],
    ]
    
    details_table = Table(details_data, colWidths=[2*inch, 4*inch])
    details_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2d3748')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
    ]))
    
    story.append(details_table)
    story.append(PageBreak())
    
    # Table of Contents with  Formatting
    story.append(Paragraph("TABLE OF CONTENTS", heading1_style))
    story.append(Spacer(1, 0.2*inch))
    
    toc_data = [
        ['Section', 'Content', 'Page'],
        ['1.', 'Executive Summary & Methodology', '3'],
        ['2.', 'Market Research & Technical Analysis', '4'],
        ['3.', 'Innovation Concepts & Development', '8'],
        ['4.', 'Selected Concept Analysis', '12'],
        ['5.', 'Technical Specifications & Architecture', '16'],
        ['6.', 'Testing & Validation Framework', '24'],
        ['7.', 'Strategic Recommendations & Next Steps', '28'],
        ['Appendix A', 'Research Sources & References', '30'],
        ['Appendix B', 'Technical Appendices', '32']
    ]
    
    toc_table = Table(toc_data, colWidths=[1*inch, 4.5*inch, 0.8*inch])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#a0aec0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')])
    ]))
    
    story.append(toc_table)
    story.append(PageBreak())
    
    # Process each section with  formatting
    for file_path, section_title in output_files:
        if os.path.exists(file_path):
            story.append(Paragraph(section_title.upper(), heading1_style))
            story.append(Spacer(1, 0.2*inch))
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            #  markdown processing
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.1*inch))
                elif line.startswith('# '):
                    story.append(Paragraph(line[2:], heading1_style))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], heading2_style))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], heading3_style))
                elif line.startswith('**') and line.endswith('**'):
                    story.append(Paragraph(f"<b>{line[2:-2]}</b>", emphasis_style))
                elif line.startswith('- ') or line.startswith('* '):
                    story.append(Paragraph(f"• {line[2:]}", body_style))
                else:
                    story.append(Paragraph(line, body_style))
            
            story.append(PageBreak())
    
    
    
    
    # Include concept visualization
    if os.path.exists("output/06_concept_visualization.png"):
        story.append(PageBreak())
        story.append(Paragraph("CONCEPT VISUALIZATION", heading2_style))
        story.append(Spacer(1, 0.2*inch))
        
        img = RLImage("output/06_concept_visualization.png", width=6*inch, height=6*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(
            "Professional concept visualization generated using advanced AI image synthesis, "
            "representing the technical specifications and design requirements in photorealistic quality.",
            ParagraphStyle('Caption', parent=styles['Normal'], 
                         fontSize=11, alignment=TA_CENTER,
                         textColor=colors.HexColor('#4a5568'),
                         fontName='Helvetica-Oblique')
        ))
    
    # Strategic Recommendations Section
    story.append(PageBreak())
    story.append(Paragraph("STRATEGIC RECOMMENDATIONS & NEXT STEPS", heading1_style))
    story.append(Spacer(1, 0.2*inch))
    
    recommendations = [
        "Immediate Actions Required",
        "• Review and validate all technical specifications with engineering teams",
        "• Initiate prototype development based on selected concept analysis", 
        "• Establish strategic partnerships identified in market research",
        "• Begin intellectual property protection processes for innovations",
        "",
        "Short-term Initiatives (3-6 months)",
        "• Complete proof-of-concept development and initial testing",
        "• Conduct market validation studies with target customer segments",
        "• Finalize manufacturing partnerships and supply chain agreements",
        "• Develop comprehensive go-to-market strategy",
        "",
        "Long-term Strategic Objectives (6-18 months)",
        "• Scale production capabilities based on market demand analysis",
        "• Expand into additional market segments identified in research",
        "• Develop next-generation enhancements and feature sets",
        "• Establish market leadership position in identified opportunity areas"
    ]
    
    for rec in recommendations:
        if rec.startswith('•'):
            story.append(Paragraph(rec, body_style))
        elif rec == "":
            story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph(rec, heading3_style))
    
    # Footer with professional closing
    story.append(Spacer(1, 0.5*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#3182ce')))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(
        "This comprehensive analysis represents the culmination of advanced multi-agent AI research "
        "and strategic planning. All recommendations are based on rigorous data analysis and "
        "industry best practices for successful product development and market entry.",
        ParagraphStyle('Closing', parent=styles['Normal'], 
                     fontSize=10, alignment=TA_CENTER,
                     textColor=colors.HexColor('#4a5568'),
                     fontName='Helvetica-Oblique')
    ))
    
    # Build the  PDF
    doc.build(story)
    
    return "output/07_comprehensive_report.pdf"


# Agent Creation Functions with  Capabilities
def create_handoff_tool(*, agent_name: str, description: str) -> callable:
    """Create a handoff tool for agent coordination."""
    tool_name = f"transfer_to_{agent_name}"
    
    def handoff_func(request: str) -> str:
        return f"HANDOFF:{agent_name}:{request}"
    
    handoff_func.__name__ = tool_name
    handoff_func.__doc__ = description
    
    return tool(handoff_func)

@traceable
def create_research_agent_node(state: RnDState) -> Command[Literal["supervisor"]]:
    """ Research Agent with comprehensive analysis."""
    logger.info("🔬 Research Agent: Conducting comprehensive research")
    
    # Use higher temperature and increased max tokens for comprehensive analysis
    # Get experiment configuration
    model_name, temperature = get_llm_config()
    
    # Use configuration
    model = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [
        get_current_datetime, 
        tavily_search,
        market_intelligence_search, 
        arxiv_research_search,
        save_research_brief
    ]
    
    workflow_context = "comprehensive new product development" if state['workflow_type'] == "new_product" else "strategic product improvement and optimization"
    
    prompt = f"""You are a Senior R&D Research Analyst with expertise in market intelligence, competitive analysis, and technical research. You are conducting a comprehensive analysis for {workflow_context}.

PROJECT BRIEF:
- Request: {state['request']}
- Workflow Type: {state['workflow_type']}
- Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
- Analysis Depth: Comprehensive (Executive-Level)

RESEARCH METHODOLOGY:
Your analysis must be thorough, strategic, and actionable. You have access to multiple research tools and must use them comprehensively:

1. **Web Search** - Use tavily_search for:
   - General web search
   - Latest news and updates
   - Industry trends and dynamics
   - Key player analysis and market positioning

2. **Market Intelligence Analysis** - Use market_intelligence_search for:
   - Competitive landscape mapping
   - Market size and growth projections
   - Industry trends and dynamics
   - Key player analysis and market positioning

3. **Academic Research Investigation** - Use academic_research_search for:
   - Peer-reviewed technical insights
   - Scientific validation of approaches
   - Academic trends and breakthroughs
   - Research gaps and opportunities

4. **Cutting-Edge Research** - Use arxiv_research_search for:
   - Latest scientific developments
   - Emerging technologies and methodologies
   - Preprint insights and future directions
   - Technical feasibility validation

COMPREHENSIVE DELIVERABLE REQUIREMENTS:
Create a detailed, executive-quality research brief in markdown format. Your analysis should be strategic, comprehensive, and actionable:

```markdown
# Comprehensive Market Research & Technical Analysis Brief

## Project Context

## Web Search Results

## Market Intelligence Analysis

## Academic Research Investigation

## Cutting-Edge Research

## Strategic Analysis

## Summary
[1-2 paragraph summary highlighting key findings, strategic opportunities, and critical insights that inform decision-making]
```

Execute your research systematically:
1. Begin with comprehensive web searches
2. Conduct comprehensive market intelligence searches
3. Search ArXiv for latest developments
4. Synthesize all findings into the strategic research brief

Save your comprehensive analysis using save_research_brief tool."""

    agent = create_react_agent(model, tools)
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content="RESEARCH_COMPLETE: Comprehensive market research and technical analysis done.")],
            "research_brief_path": "output/01_research_brief.md"
        }
    )

@traceable
def create_ideation_agent_node(state: RnDState) -> Command[Literal["supervisor"]]:
    """ Ideation Agent with advanced concept development."""
    logger.info("💡 Ideation Agent: Generating innovative concepts")
    
    # Higher temperature for  creativity, increased tokens for comprehensive analysis
    # Get experiment configuration
    model_name, temperature = get_llm_config()
    
    # Use configuration
    model = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [save_idea_concepts]
    
    # Read research brief if available
    research_content = ""
    if os.path.exists("output/01_research_brief.md"):
        with open("output/01_research_brief.md", "r", encoding="utf-8") as f:
            research_content = f.read()[:4000]  # Increased context limit
    
    workflow_context = "breakthrough innovative concepts" if state['workflow_type'] == "new_product" else "improvment_concepts"
    
    prompt = f"""You are a Senior Innovation Strategist and Design Thinking Expert responsible for developing {workflow_context} based on comprehensive market research and technical analysis.

PROJECT CONTEXT:
- Request: {state['request']}
- Workflow Type: {state['workflow_type']}
- Research Foundation: {research_content}

INNOVATION METHODOLOGY:
Apply advanced design thinking, TRIZ innovation principles, and strategic foresight to generate breakthrough concepts. Your approach should incorporate:

1. **Research-Driven Innovation**: Base concepts on market research findings and technical insights
2. **Disruptive Thinking**: Challenge conventional approaches and explore breakthrough possibilities
3. **Strategic Alignment**: Ensure concepts align with identified market opportunities
4. **Technical Feasibility**: Balance innovation with practical implementation possibilities
5. **Competitive Differentiation**: Develop unique value propositions and competitive advantages

COMPREHENSIVE DELIVERABLE REQUIREMENTS:
Generate 5 strategically distinct, professionally viable concepts with comprehensive analysis:

```markdown
# Advanced Product Concept Development & Innovation Strategy

## Innovation Framework & Methodology
[Detailed explanation of the innovation approach, design thinking process, and strategic framework employed]

## Strategic Context & Market Foundation
[How concepts align with research findings, market opportunities, and competitive landscape]

## Comprehensive Concept Portfolio

### Concept 1: [Strategic Name - Innovative & Memorable]
**Core Innovation Breakthrough:** [Primary innovative breakthrough and technological advancement]

**Strategic Value Proposition:** [Comprehensive value proposition addressing market needs and competitive advantages]

**Technical Innovation Architecture:** [Detailed technical approach, architecture, and key innovations]

**Market Positioning & Strategy:** [Target market positioning, go-to-market approach, and strategic advantages]

**Competitive Differentiation:** [Unique competitive advantages and market differentiation factors]

**Implementation Feasibility:** [Comprehensive feasibility assessment including technical, market, and resource considerations]

**Innovation Risk Assessment:** [Risk analysis and mitigation strategies]

**Market Impact Potential:** [Expected market impact, disruption potential, and growth projections]

**Strategic Fit Analysis:** [Alignment with organizational capabilities and strategic objectives]

### Concept 2: [Strategic Name]
[Complete analysis following the same comprehensive structure]

### Concept 3: [Strategic Name]
[Complete analysis following the same comprehensive structure]

### Concept 4: [Strategic Name]
[Complete analysis following the same comprehensive structure]

### Concept 5: [Strategic Name]
[Complete analysis following the same comprehensive structure]

## Strategic Concept Analysis & Comparison

### Innovation Impact Matrix
| Concept | Breakthrough Level | Market Disruption | Technical Complexity | Competitive Advantage | Implementation Timeline | Resource Requirements |
|---------|-------------------|-------------------|---------------------|----------------------|------------------------|---------------------|
| Concept 1 | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Timeline] | [Detailed Requirements] |
| Concept 2 | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Timeline] | [Detailed Requirements] |
| Concept 3 | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Timeline] | [Detailed Requirements] |
| Concept 4 | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Timeline] | [Detailed Requirements] |
| Concept 5 | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Rating] | [Detailed Timeline] | [Detailed Requirements] |

### Strategic Prioritization Framework
[Comprehensive framework for concept evaluation and selection]

### Market Opportunity Assessment
[Detailed analysis of market opportunities for each concept]

### Technology Roadmap Considerations
[Technology development pathways and strategic considerations]

## Strategic Recommendations for Concept Selection
[Top 3 concepts with detailed strategic rationale, implementation considerations, and selection criteria]

### Recommended Concept Selection Criteria
[Detailed criteria framework for stakeholder decision-making]

### Strategic Implementation Considerations
[Key factors for successful concept development and market entry]
```

Your concepts should demonstrate:
- Strategic depth and market insight
- Technical innovation and feasibility
- Competitive differentiation and advantage
- Clear value propositions and business potential
- Comprehensive risk assessment and mitigation

Generate concepts that are both visionary and practical, pushing boundaries while maintaining commercial viability.

Use save_idea_concepts tool to save your complete strategic analysis."""

    agent = create_react_agent(model, tools)
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content="IDEATION_COMPLETE: Advanced concept development and strategic innovation analysis completed.")],
            "idea_list_path": "output/02_idea_concepts.md"
        }
    )

@traceable
def create_evaluation_agent_node(state: RnDState) -> Command[Literal["supervisor"]]:
    """ Evaluation Agent with comprehensive concept analysis."""
    logger.info("🤔 Evaluation Agent: Facilitating strategic concept selection")
    
    # Get experiment configuration
    model_name, temperature = get_llm_config()
    
    # Use configuration
    model = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [human_concept_selection, save_selected_concept]
    
    # Read concepts file
    concepts_content = ""
    if os.path.exists("output/02_idea_concepts.md"):
        with open("output/02_idea_concepts.md", "r", encoding="utf-8") as f:
            concepts_content = f.read()
    
        prompt = f"""You are a Senior Strategic Consultant and Product Strategy Expert facilitating comprehensive concept evaluation and selection.

OBJECTIVE: Facilitate professional concept evaluation and selection, then create comprehensive strategic analysis of the chosen concept.

EVALUATION PROCESS:
1. Present all developed concepts to stakeholders for informed selection
2. Conduct comprehensive analysis of the selected concept
3. Develop strategic implementation framework

CONCEPTS TO PRESENT:
{concepts_content}

After facilitating selection using human_concept_selection tool, create a comprehensive strategic analysis using save_selected_concept tool with this  structure:

```markdown
# Selected Concept: Comprehensive Strategic Analysis

## Executive Summary

## Technical Innovation Deep Dive

### Core Technology Innovation
[Comprehensive analysis of the technical innovation and breakthrough aspects]

### Technical Architecture Overview
[High-level technical architecture and system design approach]

### Technology Development Pathway
[Development roadmap and technical milestones]

### Target Market Analysis
[Detailed target market definition, sizing, and segmentation]

### Market Entry Strategy
[Strategic approach to market entry and customer acquisition]

### Competitive Strategy & Positioning
[Competitive strategy and market positioning approach]


### Resource Requirements & Strategic Partnerships
[Required resources, capabilities, and strategic partnership opportunities]

### Timeline & Critical Path Analysis
[Implementation timeline with critical path and dependency analysis]

### Success Metrics & KPIs
[Key performance indicators and success measurement framework]

## Risk Assessment & Strategic Mitigation
### Strategic Risk Analysis
[Comprehensive risk assessment including market, technical, competitive, and operational risks]


## Financial & Business Model Considerations
### Revenue Model Framework
[Business model and revenue generation approach]

### Investment Requirements
[Capital requirements and investment considerations]

### Market Leadership Potential
[Potential for market leadership and competitive positioning]
```

Your analysis should provide stakeholders with comprehensive strategic insights for informed decision-making and successful concept implementation."""
    
    agent = create_react_agent(model, tools)
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]}, {"state": state})
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content="EVALUATION_COMPLETE: Comprehensive concept evaluation and strategic analysis completed.")],
            "selected_idea_path": "output/03_selected_concept.md"
        }
    )

@traceable
def create_specification_agent_node(state: RnDState) -> Command[Literal["supervisor"]]:
    """ Specification Agent with comprehensive technical specification analysis."""
    logger.info("📐 Specification Agent: Developing detailed technical specifications analysis")
    
    # Get experiment configuration
    model_name, temperature = get_llm_config()
    
    # Use configuration
    model = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [save_technical_specification]
    
    # Read selected concept
    concept_content = ""
    if os.path.exists("output/03_selected_concept.md"):
        with open("output/03_selected_concept.md", "r", encoding="utf-8") as f:
            concept_content = f.read()[:3000]
    
    prompt = f"""You are a Senior Technical Architect and Systems Engineering Expert responsible for developing comprehensive technical specifications for selected concept that translate strategic concept into actionable engineering requirements.

PROJECT CONTEXT:
- Selected Concept : {concept_content}

TECHNICAL SPECIFICATION METHODOLOGY:
Apply systems engineering principles, design for manufacturing (DFM), and industry best practices to create comprehensive technical specifications that enable successful product development.

COMPREHENSIVE DELIVERABLE REQUIREMENTS:
Develop detailed technical specifications in markdown format that serve as the foundation for engineering development.

Your specifications should serve as a complete engineering foundation for successful product development, manufacturing.

Save your comprehensive technical specifications using save_technical_specification tool."""

    agent = create_react_agent(model, tools)
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content="SPECIFICATION_COMPLETE: Comprehensive technical specifications and analytics completed.")],
            "design_spec_path": "output/04_technical_specifications.md"
        }
    )

@traceable
def create_visualization_agent_node(state: RnDState) -> Command[Literal["supervisor"]]:
    """ Visualization Agent with professional concept rendering."""
    logger.info("🎨 Visualization Agent: Creating professional concept visualization")
    
    # Get experiment configuration
    model_name, temperature = get_llm_config()
    
    # Use configuration
    model = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [generate_concept_visualization]
    
    # Read technical specifications
    specs_content = ""
    if os.path.exists("output/04_technical_specifications.md"):
        with open("output/04_technical_specifications.md", "r", encoding="utf-8") as f:
            specs_content = f.read()[:3000]
    
    prompt = f"""You are a Senior Design Visualization Specialist and Creative Director responsible for creating professional, photorealistic concept visualizations that accurately represent technical specifications and strategic concepts.

PROJECT CONTEXT:
- Technical Specifications: {specs_content}

VISUALIZATION STRATEGY:
Create a comprehensive DALL-E prompt that produces an executive-quality concept visualization suitable for:
- Professional presentations to stakeholders and investors
- Marketing materials and promotional content
- Technical documentation and development guidance
- Strategic decision-making and concept validation

Generate your comprehensive visualization prompt and execute using generate_concept_visualization tool to create the professional concept rendering."""

    agent = create_react_agent(model, tools)
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content="VISUALIZATION_COMPLETE: Professional concept visualization created with executive presentation quality.")],
            "concept_image_path": "output/06_concept_visualization.png" 
        }
    )

@traceable
def create_testing_agent_node(state: RnDState) -> Command[Literal["supervisor"]]:
    """ Testing Agent with comprehensive validation planning and analytics."""
    logger.info("🧪 Testing Agent: Developing comprehensive testing framework")
    
    # Get experiment configuration
    model_name, temperature = get_llm_config()
    
    # Use configuration
    model = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [save_testing_plan]
    
    # Read technical specifications
    specs_content = ""
    if os.path.exists("output/04_technical_specifications.md"):
        with open("output/04_technical_specifications.md", "r", encoding="utf-8") as f:
            specs_content = f.read()[:3500]
    
    prompt = f"""You are a Senior Quality Assurance Engineer and Validation Expert responsible for developing comprehensive testing and validation protocols that ensure product excellence, regulatory compliance, and market success.

PROJECT CONTEXT:
- Technical Specifications: {specs_content}

DELIVERABLE REQUIREMENTS:
Create a professional testing and validation plan in markdown format:

Your testing framework should ensure comprehensive product validation, quality assurance, and successful market launch.

Save your comprehensive testing plan using save_testing_plan tool."""

    agent = create_react_agent(model, tools)
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content="TESTING_COMPLETE: Comprehensive testing framework and validation analytics completed.")],
            "test_plan_path": "output/05_testing_plan.md"
        }
    )

@traceable
def create_documentation_agent_node(state: RnDState) -> Command[Literal["supervisor"]]:
    """ Documentation Agent with comprehensive report compilation and professional formatting."""
    logger.info("📑 Documentation Agent: Compiling comprehensive executive report")
    
    # Increased tokens for comprehensive analysis and report generation
    # Get experiment configuration
    model_name, temperature = get_llm_config()
    
    # Use configuration
    model = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [compile_comprehensive_report]
    
    prompt = """You are a Senior Technical Documentation Specialist and Executive Report Compiler responsible for creating comprehensive, executive-quality R&D reports that serve as definitive project documentation.

COMPREHENSIVE REPORT OBJECTIVE:
Compile all outputs into a professional PDF report that meets the highest standards for:
- Executive stakeholder review and decision-making
- Investor presentation and funding discussions
- Strategic planning and implementation guidance
- Technical team development and engineering reference
- Marketing and business development support

REPORT QUALITY STANDARDS:
Your report must demonstrate:
1. **Executive Excellence**: C-level presentation quality with strategic insights
2. **Technical Completeness**: Comprehensive technical documentation and specifications
4. **Strategic Value**: Actionable recommendations and clear next steps
5. **Professional Formatting**: Industry-standard formatting with  visual appeal

 REPORT COMPILATION REQUIREMENTS:
1. **Comprehensive Content Integration**: Seamlessly integrate all research phases and analyses
2. ** Formatting**: Apply professional styling, typography, and layout design
3. **Strategic Synthesis**: Provide executive summary and strategic recommendations
4. **Implementation Guidance**: Clear next steps and action items for stakeholders

Execute the comprehensive report compilation using compile_comprehensive_report tool to generate the final executive-quality PDF report.

The report should serve as a complete strategic package for product development decision-making, investment planning, and implementation guidance."""

    agent = create_react_agent(model, tools)
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content="REPORT_COMPLETE: Comprehensive PDF report compiled.")],
            "final_report_path": "output/07_comprehensive_report.pdf"
        }
    )

@traceable
def supervisor_node(state: RnDState) -> Command:
    """ Strategic Supervisor Agent with improved workflow coordination."""
    if any(
        isinstance(m, AIMessage) and "REPORT_COMPLETE" in m.content
        for m in state.get("messages", [])
    ):
        return Command(goto="__end__")

    # Create handoff tools
    tools = [
        create_handoff_tool(agent_name="research_agent", description="Transfer to research agent for comprehensive market and technical analysis"),
        create_handoff_tool(agent_name="ideation_agent", description="Transfer to ideation agent for advanced concept development"),
        create_handoff_tool(agent_name="evaluation_agent", description="Transfer to evaluation agent for strategic concept selection"),
        create_handoff_tool(agent_name="specification_agent", description="Transfer to specification agent for comprehensive technical requirements"),
        create_handoff_tool(agent_name="visualization_agent", description="Transfer to visualization agent for professional concept rendering"),
        create_handoff_tool(agent_name="testing_agent", description="Transfer to testing agent for comprehensive validation planning"),
        create_handoff_tool(agent_name="documentation_agent", description="Transfer to documentation agent for executive report compilation"),
    ]
    
    # Determine workflow type if not set
    workflow_type = state.get("workflow_type")
    if not workflow_type:
        request_lower = state["request"].lower()
        improvement_keywords = ["improve", "enhance", "optimize", "upgrade", "refine", "better", "fix", "update", "modify"]
        if any(keyword in request_lower for keyword in improvement_keywords):
            workflow_type = "product_improvement"
        else:
            workflow_type = "new_product"
        logger.info(f"📋 Workflow type determined: {workflow_type}")
    
    # Determine next agent based on completed work
    completed_phases = [msg.content for msg in state['messages'] if isinstance(msg, AIMessage)]
    
    prompt = f"""You are the Strategic R&D Project Supervisor coordinating a comprehensive multi-agent workflow for professional product development.

PROJECT OVERVIEW:
- Request: {state['request']}
- Workflow Type: {workflow_type}
- Current Iteration: {state.get('iterations', 0)}


WORKFLOW SEQUENCE:
1. research_agent → Comprehensive web search, market research.
2. ideation_agent → Advanced concept development with strategic innovation framework
3. evaluation_agent → Strategic concept selection and comprehensive analysis
4. specification_agent → Detailed technical specifications with performance analytics
5. visualization_agent → Professional concept visualization and rendering
6. testing_agent → Comprehensive testing framework with validation analytics
7. documentation_agent → Executive-quality comprehensive report compilation
8. END

# COMPLETED PHASES: {completed_phases}

STRATEGIC COORDINATION TASK:
Analyze the current project state and transfer control to the next appropriate agent using the corresponding transfer_to_[agent_name] tool only if the previous phase is complete.

Ensure each phase builds upon previous work to create a comprehensive, professional R&D analysis suitable for executive decision-making and strategic implementation.

"""
    
    # Get experiment configuration
    model_name, temperature = get_llm_config()
    
    # Use configuration
    model = ChatOpenAI(model=model_name, temperature=temperature)
    supervisor = create_react_agent(model, tools)
    
    result = supervisor.invoke({"messages": [HumanMessage(content=prompt)]})
    
    # Process handoff commands
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.content and msg.content.startswith("HANDOFF:"):
            parts = msg.content.split(":", 2)
            if len(parts) >= 3:
                _, target_agent, request = parts
                return Command(
                    goto=target_agent,
                    update={
                        "messages": [AIMessage(content=f"Transferring to {target_agent} for  analysis")],
                        "iterations": state.get("iterations", 0) + 1,
                        "workflow_type": workflow_type
                    }
                )
    
    
    
    # Default: stay in supervisor
    return Command(
        goto="supervisor",
        update={
            "iterations": state.get("iterations", 0) + 1,
            "workflow_type": workflow_type
        }
    )

    
def build_rd_workflow_graph():
    """Build the R&D multi-agent workflow graph with 2025 LangGraph standards."""
    # Use proper StateGraph with START and END constants
    workflow = StateGraph(RnDState)
    
    # Add all agent nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", create_research_agent_node)
    workflow.add_node("ideation_agent", create_ideation_agent_node)
    workflow.add_node("evaluation_agent", create_evaluation_agent_node)
    workflow.add_node("specification_agent", create_specification_agent_node)
    workflow.add_node("visualization_agent", create_visualization_agent_node)
    workflow.add_node("testing_agent", create_testing_agent_node)
    workflow.add_node("documentation_agent", create_documentation_agent_node)
    
    # Set entry point
    workflow.add_edge(START, "supervisor")

# Add conditional edges from supervisor to show all possible paths
    


    # Add edges back to supervisor
    for agent in ["research_agent", "ideation_agent", "evaluation_agent", 
                  "specification_agent", "visualization_agent", "testing_agent", "documentation_agent"]:
        workflow.add_edge(agent, "supervisor")

    # Documentation agent goes to END
    workflow.add_edge("supervisor", END)
        
    # Compile with memory
    #checkpointer = MemorySaver()
    #return workflow.compile(checkpointer=checkpointer)

    #Compile for LangGraph Studio (command:langgraph dev or python main.py --evaluate) 
    return workflow.compile()

def visualize_workflow_graph():
    """Generate and save workflow graph visualization using 2025 LangGraph standards."""
    logger.info("📊 Generating workflow graph visualization")
    
    try:
        # Build graph using proper patterns
        graph = build_rd_workflow_graph()
        
        # Create output directory
        Path("output").mkdir(exist_ok=True)
        
        # Use proper method call for 2025 standards
        try:
            # Primary method: use get_graph().draw_mermaid_png()
            mermaid_png = graph.get_graph().draw_mermaid_png()
            
            with open("output/workflow_graph.png", "wb") as f:
                f.write(mermaid_png)
                
            console.print("[green]✅  workflow graph saved to output/workflow_graph.png[/green]")
            
        except Exception as e:
            logger.warning(f"Primary visualization method failed: {e}")
            
            # Fallback method: generate mermaid text and save
            try:
                mermaid_text = graph.get_graph().draw_mermaid()
                
                with open("output/workflow_graph.mermaid", "w") as f:
                    f.write(mermaid_text)
                    
                console.print("[yellow]⚠️ Generated mermaid text file: output/workflow_graph.mermaid[/yellow]")
                console.print("[dim]You can visualize this at: https://mermaid.live/[/dim]")
                
            except Exception as e2:
                logger.error(f"Fallback visualization failed: {e2}")
                console.print("[red]❌ Could not generate graph visualization[/red]")
                return None
        
        # Display the image if successfully generated
        try:
            if os.path.exists("output/workflow_graph.png"):
                image = Image.open("output/workflow_graph.png")
                image.show()
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not display image: {e}[/yellow]")
            
        return "output/workflow_graph.png"
        
    except Exception as e:
        logger.error(f"Graph visualization error: {e}")
        console.print(f"[red]❌ Failed to generate graph visualization: {e}[/red]")
        return None


@traceable
def run_rd_workflow(request: str):
    """Execute the  comprehensive R&D workflow."""
    graph = build_rd_workflow_graph()
    
    # Initialize  workflow state
    initial_state = {
        "messages": [HumanMessage(content=request)],
        "request": request,
        "workflow_type": None,  # Will be determined by supervisor
        "iterations": 0,
        "current_agent": "supervisor"
    }
    
    # Configuration for workflow execution
    config = {"configurable": {"thread_id": f"-rd-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"},"recursion_limit": 100}
    
    console.print(f"\n[bold cyan]🚀 INITIATING  R&D WORKFLOW[/bold cyan]")
    console.print(f"[dim]Request: {request}[/dim]")
    console.print(f"[dim]Session ID: {config['configurable']['thread_id']}[/dim]")
    
    try:
        # Stream  workflow execution
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            for node, values in event.items():
                if values and isinstance(values, dict) and "messages" in values:
                    for msg in values["messages"]:
                        if isinstance(msg, AIMessage):
                            console.print(f"[green]✓[/green] {node}: {msg.content}")
        
        console.print(f"\n[bold green]🎉  R&D WORKFLOW COMPLETED SUCCESSFULLY[/bold green]")
        console.print("\n[cyan]📂 Generated Comprehensive Outputs:[/cyan]")
        
        # List generated files with  formatting
        output_files = [
            ("01_research_brief.md", "Web Search, Market Research & Technical Analysis"),
            ("02_idea_concepts.md", "Innovation Concepts & Strategic Development"), 
            ("03_selected_concept.md", "Selected Concept Strategic Analysis"),
            ("04_technical_specifications.md", "Comprehensive Technical Specifications"),
            ("05_testing_plan.md", "Testing & Validation Framework"),
            ("06_concept_visualization.png", "Professional Concept Visualization"),
            ("07_comprehensive_report.pdf", "Executive Comprehensive Report")
        ]
        
        for file, description in output_files:
            file_path = f"output/{file}"
            if os.path.exists(file_path):
                console.print(f"[green]✓[/green] {file_path} - {description}")
            else:
                console.print(f"[yellow]⚠[/yellow] {file_path} - {description} (not generated)")
        
        
        console.print(f"\n[bold]📖 Review the executive comprehensive report:[/bold]")
        console.print(f"[bold cyan]    → output/07_comprehensive_report.pdf[/bold cyan]")
        console.print(f"[dim]This report includes all analyses and strategic recommendations[/dim]")
        
    except Exception as e:
        logger.error(f" workflow execution error: {e}")
        console.print(f"[red]❌  workflow failed: {e}[/red]")
        raise

@traceable
async def run_workflow_for_evaluation(inputs: Dict[str, any]) -> Dict[str, any]:
    """
    Wrapper function to run your Agent Laboratory workflow for evaluation purposes.
    This function is called by the evaluation system.
    """
    try:
        print(f"🔬 Running evaluation workflow for: {inputs.get('request', 'Unknown request')}")
        
        # Create workflow instance
        workflow = build_rd_workflow_graph()
        
        # Prepare initial state
        initial_state = {
            "request": inputs.get("request", ""),
            "workflow_type": inputs.get("workflow_type", "new_product"),
            "messages": [HumanMessage(content=inputs.get("request", ""))],
            "iterations": 0,
            "current_agent": "supervisor"
        }
        
        # Create unique config for this run
        config = {
            "configurable": {
                "thread_id": f"eval-{uuid.uuid4().hex[:8]}",
                "recursion_limit": 100
            }
        }
        
        # Run the workflow with async invoke
        final_state = await workflow.ainvoke(initial_state, config=config)
        
        # Collect outputs from generated files
        outputs = collect_workflow_outputs()
        
        # Add final state information to outputs
        outputs.update({
            "workflow_state": {
                "final_agent": final_state.get("current_agent", "unknown"),
                "iterations": final_state.get("iterations", 0),
                "workflow_type": final_state.get("workflow_type", "unknown")
            },
            "success": True
        })
        
        return outputs
        
    except Exception as e:
        print(f"❌ Error in evaluation workflow: {e}")
        import traceback
        traceback.print_exc()
        return {
            "final_report": f"Error: {str(e)}",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

        
def collect_workflow_outputs() -> Dict[str, any]:
    """
    Collect and return outputs from your Agent Laboratory workflow.
     to match the new evaluation expectations.
    """
    outputs = {
        "generated_files": []
    }
    
    # Updated output file mapping to match evaluators
    output_files = {
        "research_brief": "output/01_research_brief.md",
        "concepts": "output/02_idea_concepts.md",          
        "selected_concept": "output/03_selected_concept.md",
        "specifications": "output/04_technical_specifications.md",
        "validation_plan": "output/05_testing_plan.md",    
        "visualization": "output/06_concept_visualization.png",
        "final_report": "output/07_comprehensive_report.pdf"
    }
    
    for output_type, file_path in output_files.items():
        if os.path.exists(file_path):
            outputs["generated_files"].append(file_path)
            
            # Read text content for evaluation
            if file_path.endswith('.md'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        outputs[output_type] = content  # Full content for agent evaluations
                except Exception as e:
                    outputs[output_type] = f"Error reading {file_path}: {e}"
            elif file_path.endswith('.pdf'):
                outputs[output_type] = f"PDF report generated at {file_path}"
            elif file_path.endswith('.png'):
                outputs[output_type] = f"Visualization generated at {file_path}"
    
    return outputs

async def run_evaluation_suite():
    """
    Run the complete evaluation suite on your Agent Laboratory.
    This function imports and runs the  evaluation system.
    """
    try:
        print("🚀 Starting  Agent Laboratory Evaluation Suite...")
        print("📊 Now with per-agent quality metrics!")
        
        # Import  evaluation system
        from agent_lab_evals import AgentLabEvaluationSuite
        
        # Initialize evaluation suite
        eval_suite = AgentLabEvaluationSuite()
        
        # Run comprehensive evaluation using your workflow
        results = await eval_suite.run_evaluation(run_workflow_for_evaluation)
        
        print("✅  evaluation completed successfully!")
        
        # Display per-agent results
        if "average_agent_scores" in results:
            print("\n📊 Per-Agent Quality Scores:")
            for agent, score in results["average_agent_scores"].items():
                print(f"  {agent.capitalize()} Agent: {score:.2%}")
        
        return results
        
    except ImportError:
        print("❌ Error: agent_lab_evals.py not found.")
        print("📥 Make sure you have the evaluation file in your directory.")
        return None
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return None

def setup_langsmith_tracing():
    """
    Setup LangSmith tracing for your Agent Laboratory.
    Call this function at the start of your main workflow.
    """
    import os
    
    # Check if LangSmith environment variables are set
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("⚠️  LangSmith API key not set. Evaluation features will be limited.")
        print("   Set LANGSMITH_API_KEY in your .env file for full functionality.")
        return False
    
    # Enable tracing
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    
    print("✅ LangSmith tracing enabled")
    return True

async def main_workflow():
    """ main workflow with evaluation integration."""
    
    # Setup tracing
    setup_langsmith_tracing()
    
    # Check for evaluation flag
    if "--evaluate" in sys.argv:
        print("🔬 Running evaluation mode...")
        results = await run_evaluation_suite()
        return results
    
    # Original workflow execution
    print("🔬  Virtual AI Laboratory for R&D")
    
    try:
        # Your existing workflow code
        user_request = input("\nWhat would you like to develop or improve?? \n> ")
        
        if not user_request.strip():
            print("❌ Please provide a valid request.")
            return
        
        workflow = build_rd_workflow_graph()
        
        initial_state = {
            "request": user_request,
            "messages": [],
            "iterations": 0
        }
        
        print("\n✅ Workflow initiated...")
        
        # Run with tracing enabled
        config = {"configurable": {"thread_id": f"session_{uuid.uuid4().hex[:8]}"}}
        final_state = await workflow.ainvoke(initial_state, config=config)


        print("\n🎉 Workflow completed successfully!")
        print("\n📖 Review the comprehensive outputs:")
        
        # Your existing output display code
        output_files = [
            ("output/01_research_brief.md", "🔍 Market Research & Technical Analysis Brief"),
            ("output/02_innovation_concepts.md", "💡 Innovation Concepts Portfolio"),
            ("output/03_selected_concept.md", "⚖️ Selected Concept Analysis"),
            ("output/04_technical_specifications.md", "📋 Technical Specifications"),
            ("output/05_test_validation_plan.md", "🧪 Testing & Validation Plan"),
            ("output/06_concept_visualization.png", "🎨 Concept Visualization"),
            ("output/07_comprehensive_report.pdf", "📄 Executive Comprehensive Report")
        ]
        
        for file_path, description in output_files:
            if os.path.exists(file_path):
                print(f"    → {description}: {file_path}")
        
        # Display final state information
        print(f"\n📊 Workflow Statistics:")
        print(f"    • Total iterations: {final_state.get('iterations', 0)}")
        print(f"    • Messages processed: {len(final_state.get('messages', []))}")
        print(f"    • Workflow type: {final_state.get('workflow_type', 'Unknown')}")
        
    except Exception as e:
        print(f"\n❌ Error in workflow execution: {e}")
        print("🔧 Please check your configuration and try again.")


#Integration with LangGraph
app = build_rd_workflow_graph()

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if "--evaluate" in sys.argv:
            # Run evaluation
            asyncio.run(main_workflow())
        elif "--graph" in sys.argv:
            # Generate graph visualization (your existing code)
            visualize_workflow_graph()
        elif "--help" in sys.argv:
            print("Agent Laboratory - Multi-Agent R&D Workflow System")
            print("\nUsage:")
            print("  python main.py                 # Run interactive workflow")
            print("  python main.py --evaluate      # Run evaluation suite") 
            print("  python main.py --graph         # Generate workflow graph")
            print("  python main.py --help          # Show this help")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Run normal interactive workflow
        asyncio.run(main_workflow())
