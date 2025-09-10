from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import dotenv
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import json
import uvicorn
import numpy as np
import re

# LangChain imports
from langchain_experimental.agents import create_pandas_dataframe_agent
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
dotenv.load_dotenv()

app = FastAPI(
    title="INGRES Groundwater API",
    description="AI-powered groundwater data analysis API",
    version="1.0.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change * to your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class DataSummary(BaseModel):
    total_records: int
    states_count: int
    years_range: tuple
    available_years: List[int]
    available_states: List[str]

class QueryResponse(BaseModel):
    response: str
    data: Optional[List[Dict]] = None
    chart: Optional[Dict] = None
    chart_type: Optional[str] = None

class GroundwaterAssistant:
    """Groundwater Assistant using PandasAI Agent"""
    
    def __init__(self, csv_path: str):
        """Initialize the assistant with CSV data and PandasAI agent"""
        self.df = self.load_data(csv_path)
        self.agent = None
        self.langchain_agent = None
        self.setup_pandas_ai_agent()
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess the CSV data"""
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(csv_path)
        
        # Basic data cleaning - handle null values properly
        df = df.dropna(subset=['State'], how='all')  # Only drop if State is completely null
        
        # Convert Year to numeric, keep NaN for missing years
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # Handle numeric columns - convert to numeric but keep NaN
        numeric_columns = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean State names - remove extra spaces and standardize
        df['State'] = df['State'].str.strip()
        
        # Clean Stage_of_Extraction values
        if 'Stage_of_Extraction' in df.columns:
            df['Stage_of_Extraction'] = df['Stage_of_Extraction'].str.strip()
        
        # Add debug logging
        print(f"[DEBUG] Data loaded: {len(df)} records")
        print(f"[DEBUG] States: {df['State'].unique()[:10]}")  # Show first 10 states
        print(f"[DEBUG] Year range: {df['Year'].min()} to {df['Year'].max()}")
        print(f"[DEBUG] Stage extraction values: {df['Stage_of_Extraction'].unique()}")
        print(f"[DEBUG] Sample data structure:")
        print(df.head())
        
        return df
    
    def setup_pandas_ai_agent(self):
        """Setup PandasAI agent with Groq"""
        try:
            load_dotenv()
            groq_key = os.getenv("GROQ_API_KEY")
            
            if not groq_key or not groq_key.startswith("gsk_"):
                print("[DEBUG] GROQ API key not found. Please set GROQ_API_KEY in environment")
                self.langchain_agent = None
                return
            
            llm_groq = ChatGroq(
                groq_api_key=groq_key,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0,
            )

            self.langchain_agent = create_pandas_dataframe_agent(
                llm_groq,
                self.df,
                verbose=False,
                return_intermediate_steps=False,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
            print("[DEBUG] LangChain agent initialized successfully")

        except Exception as e:
            print(f"[DEBUG] Error setting up AI agent: {str(e)}")
            self.langchain_agent = None

    def extract_states_from_query(self, query: str) -> List[str]:
        """Extract state names from query with better matching"""
        query_lower = query.lower()
        states = self.df['State'].dropna().unique()
        mentioned_states = []
        
        for state in states:
            if pd.notna(state):
                # Try exact match first
                if state.lower() in query_lower:
                    mentioned_states.append(state)
                # Try partial matching for common abbreviations
                elif len(state.split()) > 1:  # Multi-word states
                    words = state.lower().split()
                    if any(word in query_lower for word in words if len(word) > 3):
                        mentioned_states.append(state)
            # Handle common state name variations (with word-boundary matching for abbreviations)
            state_variations = {
                'jammu kashmir': 'Jammu and Kashmir',
                'j&k': 'Jammu and Kashmir', 
                'jk': 'Jammu and Kashmir',
                'andhra pradesh': 'Andhra Pradesh',
                'ap': 'Andhra Pradesh',
                'uttar pradesh': 'Uttar Pradesh',
                'up': 'Uttar Pradesh',
                'madhya pradesh': 'Madhya Pradesh',
                'mp': 'Madhya Pradesh'
            }

            for variation, full_name in state_variations.items():
                try:
                    pattern = rf"\b{re.escape(variation)}\b"
                    if re.search(pattern, query_lower) and full_name in states and full_name not in mentioned_states:
                        mentioned_states.append(full_name)
                except Exception:
                    if variation == query_lower.strip() and full_name in states and full_name not in mentioned_states:
                        mentioned_states.append(full_name)
        
        print(f"[DEBUG] Extracted states from query '{query}': {mentioned_states}")
        return mentioned_states
    
    def extract_year_from_query(self, query: str) -> Optional[int]:
        """Extract year from query"""
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        year = int(year_match.group()) if year_match else None
        print(f"[DEBUG] Extracted year from query '{query}': {year}")
        # Try to extract year from common phrases like "in 2019"
        if not year:
            year_phrase_match = re.search(r'\bin\s+((?:19|20)\d{2})\b', query.lower())
            if year_phrase_match:
                try:
                    year = int(year_phrase_match.group(1))
                except Exception:
                    year = None
        return year
    
    def parse_comparison_query(self, query: str) -> dict:
        """Parse comparison queries to extract entities and filters"""
        query_lower = query.lower()
        
        # Extract comparison type
        comparison_type = None
        if 'stage' in query_lower and ('extraction' in query_lower or 'exploit' in query_lower):
            comparison_type = 'stage_extraction'
        elif 'groundwater' in query_lower and 'level' in query_lower:
            comparison_type = 'groundwater_level'
        elif 'recharge' in query_lower:
            comparison_type = 'recharge'
        
        # Extract entities
        states = self.extract_states_from_query(query)
        year = self.extract_year_from_query(query)
        
        return {
            'comparison_type': comparison_type,
            'states': states,
            'year': year,
            'is_comparison': any(word in query_lower for word in ['compare', 'comparison', 'vs', 'between'])
        }

    def query_data(self, user_question: str) -> Dict[str, Any]:
        """Process user query using Groq LangChain agent"""
        print(f"[DEBUG] Processing query: {user_question}")
        
        try:
            # Handle greeting queries
            if self.is_greeting_query(user_question):
                return {
                    "response": self.handle_greeting_query(user_question),
                    "data": None,
                    "chart": None,
                    "chart_type": None
                }
            
            # Handle specific data queries with direct pandas calculations
            direct_result = self.handle_direct_query(user_question)
            if direct_result:
                relevant_data = self.extract_relevant_data(user_question)
                chart_data, chart_type = self.create_visualization_json(user_question, relevant_data)
                
                print(f"[DEBUG] Direct result: {direct_result}")
                print(f"[DEBUG] Chart type: {chart_type}")
                
                return {
                    "response": direct_result,
                    "data": None,
                    "chart": chart_data,
                    "chart_type": chart_type
                }
            
            # For complex queries, use the agent if available
            if self.langchain_agent:
                enhanced_question = f"""
                {user_question}
                
                Instructions:
                - Provide a clear, conversational response
                - For numerical results, include the metric name and value
                - Keep responses concise and natural
                - Don't mention dataset details, dataframe information, or data counts
                - Don't show code or intermediate steps
                - Handle missing data gracefully by saying "no data available" when needed
                """

                response = self.langchain_agent.run(enhanced_question)
                response = self.clean_response(response)
                
                # Extract relevant data for visualization
                relevant_data = self.extract_relevant_data(user_question)
                chart_data, chart_type = self.create_visualization_json(user_question, relevant_data)

                print(f"[DEBUG] Agent response: {response}")
                print(f"[DEBUG] Chart type: {chart_type}")

                return {
                    "response": str(response),
                    "data": None,
                    "chart": chart_data,
                    "chart_type": chart_type
                }
            else:
                # Fallback when AI agent is not configured: still try to be helpful
                relevant_data = self.extract_relevant_data(user_question)
                chart_data, chart_type = self.create_visualization_json(user_question, relevant_data)
                
                # Try to provide a basic response based on the data
                if relevant_data is not None and not relevant_data.empty:
                    response = f"I found {len(relevant_data)} records matching your query. Here's the data visualization."
                else:
                    response = "I couldn't find specific data for your query, but I can help you explore the groundwater dataset. Try asking about specific states, years, or metrics like groundwater levels, recharge, or extraction."
                
                return {
                    "response": response,
                    "data": relevant_data.to_dict('records') if relevant_data is not None else None,
                    "chart": chart_data,
                    "chart_type": chart_type
                }

        except Exception as e:
            print(f"[DEBUG] Error processing query: {str(e)}")
            return {
                "response": "I'm having trouble processing that question. Could you please rephrase it or ask something more specific about the groundwater data?",
                "data": None,
                "chart": None,
                "chart_type": None
            }

    def handle_direct_query(self, query: str) -> Optional[str]:
        """Handle specific queries with direct pandas calculations"""
        try:
            query_lower = query.lower()
            
            # Extract state and year from query
            mentioned_states = self.extract_states_from_query(query)
            mentioned_year = self.extract_year_from_query(query)
            
            # Filter data based on query
            filtered_df = self.df.copy()
            if mentioned_states:
                filtered_df = filtered_df[filtered_df['State'].isin(mentioned_states)]
            if mentioned_year:
                filtered_df = filtered_df[filtered_df['Year'] == mentioned_year]
            
            print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")
            print(f"[DEBUG] Filtered data preview:")
            if not filtered_df.empty:
                print(filtered_df[['State', 'Year', 'Stage_of_Extraction']].head())
            
            if filtered_df.empty:
                if mentioned_year and mentioned_states:
                    return f"No data found for {', '.join(mentioned_states)} in {mentioned_year}."
                elif mentioned_states:
                    return f"No data found for {', '.join(mentioned_states)}."
                elif mentioned_year:
                    return f"No data found for {mentioned_year}."
                else:
                    return "No data found for the specified criteria."
            
            # Handle comparison queries for stage extraction
            if 'compare' in query_lower and 'stage' in query_lower and 'extraction' in query_lower:
                return self.handle_stage_comparison(filtered_df, mentioned_states, mentioned_year)
            
            # Handle specific query patterns
            if 'average' in query_lower and 'groundwater' in query_lower:
                if 'Groundwater_Level_m' in filtered_df.columns:
                    valid_data = filtered_df['Groundwater_Level_m'].dropna()
                    if not valid_data.empty:
                        avg_val = valid_data.mean()
                        location = f" in {', '.join(mentioned_states)}" if mentioned_states else ""
                        year_str = f" for {mentioned_year}" if mentioned_year else ""
                        return f"The average groundwater level{location}{year_str} is {avg_val:.1f} meters."
                    else:
                        return "No groundwater level data available for the specified criteria."
            
            if 'total' in query_lower and 'recharge' in query_lower:
                if 'Recharge_mm' in filtered_df.columns:
                    valid_data = filtered_df['Recharge_mm'].dropna()
                    if not valid_data.empty:
                        total_val = valid_data.sum()
                        location = f" in {', '.join(mentioned_states)}" if mentioned_states else ""
                        year_str = f" for {mentioned_year}" if mentioned_year else ""
                        return f"The total recharge{location}{year_str} is {total_val:.2f} mm."
                    else:
                        return "No recharge data available for the specified criteria."
            
            if 'trend' in query_lower and 'groundwater' in query_lower:
                return f"Here's the groundwater level trend analysis:"
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Error in handle_direct_query: {str(e)}")
            return None

    def handle_stage_comparison(self, filtered_df: pd.DataFrame, mentioned_states: List[str], mentioned_year: Optional[int]) -> str:
        """Handle stage extraction comparison queries with improved formatting"""
        print(f"[DEBUG] Handling stage comparison for states: {mentioned_states}, year: {mentioned_year}")
        print(f"[DEBUG] Available data in filtered_df:")
        print(filtered_df[['State', 'Year', 'Stage_of_Extraction']].to_string())
        # Add this check at the beginning of the method:
        if mentioned_year:
            year_filtered = filtered_df[filtered_df['Year'] == mentioned_year]
            if year_filtered.empty:
                return f"No stage extraction data found for {mentioned_year}."
            filtered_df = year_filtered
        if 'Stage_of_Extraction' not in filtered_df.columns:
            return "Stage extraction data is not available."
        
        comparison_results = []
        
        if mentioned_states:
            # Compare specific states
            for state in mentioned_states:
                state_data = filtered_df[filtered_df['State'] == state]
                
                if state_data.empty:
                    comparison_results.append(f"No data available for {state}")
                else:
                    # Get unique stages for this state
                    stages = state_data['Stage_of_Extraction'].dropna().unique()
                    if len(stages) > 0:
                        stage_list = [f"'{stage}'" for stage in stages]
                        if len(stage_list) == 1:
                            comparison_results.append(f"For {state}, the stages of extraction are {stage_list[0]}")
                        else:
                            comparison_results.append(f"For {state}, the stages of extraction are {', '.join(stage_list)}")
                    else:
                        comparison_results.append(f"No stage data available for {state}")
        else:
            # No specific states mentioned, show overall data
            stages = filtered_df['Stage_of_Extraction'].dropna().unique()
            if len(stages) > 0:
                stage_list = [f"'{stage}'" for stage in stages]
                year_str = f" in {mentioned_year}" if mentioned_year else ""
                comparison_results.append(f"The stages of extraction{year_str} are {', '.join(stage_list)}")
            else:
                comparison_results.append("No stage extraction data available")
        
        year_str = f" in {mentioned_year}" if mentioned_year else ""
        if mentioned_states and len(mentioned_states) > 1:
            result = f"Stage extraction comparison{year_str}:\n" + ". ".join(comparison_results)
        else:
            result = ". ".join(comparison_results)
        
        print(f"[DEBUG] Stage comparison result: {result}")
        return result
    def create_visualization_json(self, query: str, data: Optional[pd.DataFrame]) -> tuple[Optional[Dict], Optional[str]]:
        """Create visualization data in JSON format for frontend - Bar, Pie, Line charts only"""
        print(f"[DEBUG] Creating visualization for query: {query}")
        
        if data is None or data.empty:
            print("[DEBUG] No data available for visualization")
            return None, None
            
        query_lower = query.lower()
        
        try:
            # Distribution queries - PIE CHART
            if any(word in query_lower for word in ['distribution', 'breakdown', 'share', 'proportion', 'percentage']):
                return self.create_distribution_pie_chart_json(data, query_lower)
            
            # Stage extraction comparison - BAR CHART
            elif 'compare' in query_lower and 'stage' in query_lower and 'extraction' in query_lower:
                return self.create_stage_comparison_bar_chart_json(data, query_lower)
            
            # Trend analysis - LINE CHART
            elif 'trend' in query_lower and 'Year' in data.columns:
                return self.create_trend_line_chart_json(data)
            
            # General comparison queries (including groundwater level comparisons) - BAR CHART
            elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'between']):
                return self.create_comparison_bar_chart_json(data, query_lower)
            
            # State-specific average queries - BAR CHART (if multiple states) or LINE CHART (if temporal)
            elif 'average' in query_lower and 'State' in data.columns:
                return self.create_average_chart_json(data, query_lower)
            
            return None, None
                        
        except Exception as e:
            print(f"[DEBUG] Chart creation error: {str(e)}")
            return None, None 

    def create_average_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create appropriate chart for average queries - BAR for states, LINE for temporal"""
        print(f"[DEBUG] Creating average chart")
        
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        
        # Determine which metric to show based on query
        metric_col = 'Groundwater_Level_m'  # default
        if 'recharge' in query:
            metric_col = 'Recharge_mm' if 'Recharge_mm' in data.columns else metric_col
        elif 'extraction' in query:
            metric_col = 'Extraction_mm' if 'Extraction_mm' in data.columns else metric_col
        
        if metric_col not in data.columns or data[metric_col].notna().sum() == 0:
            print(f"[DEBUG] No data available for metric: {metric_col}")
            return None, None
        
        # If we have year data and multiple years, show LINE chart (temporal trend)
        if 'Year' in data.columns and data['Year'].nunique() > 1:
            clean_data = data.dropna(subset=[metric_col, 'Year'])
            yearly_avg = clean_data.groupby('Year')[metric_col].mean().reset_index()
            
            chart_data = {
                "type": "line",
                "title": f'{metric_col.replace("_", " ").title()} Average Over Years',
                "xAxis": {
                    "label": "Year",
                    "type": "category"
                },
                "yAxis": {
                    "label": metric_col.replace("_", " ").title(),
                    "type": "value"
                },
                "series": [{
                    "name": metric_col.replace("_", " ").title(),
                    "type": "line",
                    "data": [
                        {"x": int(row['Year']), "y": float(row[metric_col])} 
                        for _, row in yearly_avg.iterrows()
                    ]
                }]
            }
            
            print(f"[DEBUG] Average LINE chart created with {len(yearly_avg)} data points")
            return chart_data, "line"
        
        # If multiple states, show BAR chart (state comparison)
        elif 'State' in data.columns and data['State'].nunique() > 1:
            clean_data = data.dropna(subset=[metric_col, 'State'])
            state_avg = clean_data.groupby('State')[metric_col].mean().reset_index()
            
            chart_data = {
                "type": "bar",
                "title": f'{metric_col.replace("_", " ").title()} Average by State',
                "xAxis": {
                    "label": "State",
                    "type": "category",
                    "categories": list(state_avg['State'])
                },
                "yAxis": {
                    "label": metric_col.replace("_", " ").title(),
                    "type": "value"
                },
                "series": [{
                    "name": metric_col.replace("_", " ").title(),
                    "type": "bar", 
                    "data": [float(val) for val in state_avg[metric_col]]
                }]
            }
            
            print(f"[DEBUG] Average BAR chart created with {len(state_avg)} data points")
            return chart_data, "bar"
        
        return None, None

    def create_stage_comparison_bar_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create BAR chart for stage extraction comparison"""
        print(f"[DEBUG] Creating stage comparison BAR chart")
        print(f"[DEBUG] Input data shape: {data.shape}")
        
        if 'Stage_of_Extraction' not in data.columns:
            print("[DEBUG] No Stage_of_Extraction column found")
            return None, None
        
        # Clean the data - remove rows with null Stage_of_Extraction
        clean_data = data.dropna(subset=['Stage_of_Extraction']).copy()
        
        if clean_data.empty:
            print("[DEBUG] No valid stage extraction data after cleaning")
            return None, None
        
        # Get unique states and stages
        states = clean_data['State'].dropna().unique()
        all_stages = clean_data['Stage_of_Extraction'].dropna().unique()
        
        print(f"[DEBUG] States in chart: {states}")
        print(f"[DEBUG] Stages in chart: {all_stages}")

        # Define colors for different stages
        stage_colors = {
            'Safe': '#28a745',           # Green
            'Semi-Critical': '#ffc107',  # Yellow
            'Critical': '#fd7e14',       # Orange
            'Over-Exploited': '#dc3545', # Red
            'Over-Critical': '#dc3545'   # Red (alternative naming)
        }

        chart_data = {
            "type": "bar",
            "title": f"Stage of Extraction Comparison - {', '.join(states)}" if len(states) <= 3 else "Stage of Extraction by State",
            "xAxis": {
                "label": "State",
                "type": "category",
                "categories": list(states)
            },
            "yAxis": {
                "label": "Count of Records",
                "type": "value"
            },
            "series": []
        }

        # Create one series per stage, with data points for each state
        for stage in sorted(all_stages):
            series_data = []
            
            for state in states:
                # Count occurrences of this stage in this state
                count = len(clean_data[
                    (clean_data['State'] == state) & 
                    (clean_data['Stage_of_Extraction'] == stage)
                ])
                series_data.append(count)
            
            chart_data["series"].append({
                "name": stage,
                "type": "bar",
                "data": series_data,
                "color": stage_colors.get(stage, '#007bff')
            })
        
        print(f"[DEBUG] Stage comparison BAR chart created with {len(chart_data['series'])} series")
        return chart_data, "bar"
    
    def create_comparison_bar_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create BAR chart for general comparisons including groundwater level"""
        print(f"[DEBUG] Creating comparison BAR chart")
        print(f"[DEBUG] Query: {query}")
        
        # Determine the metric to compare based on query
        metric_col = None
        chart_title = ""
        
        if 'groundwater' in query.lower() and 'level' in query.lower():
            metric_col = 'Groundwater_Level_m'
            chart_title = "Groundwater Level Comparison"
        elif 'recharge' in query.lower():
            metric_col = 'Recharge_mm'
            chart_title = "Recharge Comparison"
        elif 'extraction' in query.lower():
            metric_col = 'Extraction_mm'
            chart_title = "Extraction Comparison"
        elif 'stage' in query.lower() and 'extraction' in query.lower():
            # Handle stage extraction separately
            return self.create_stage_comparison_bar_chart_json(data, query)
        
        # If no specific metric found, use the first available numeric column
        if not metric_col:
            numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
            available_cols = [col for col in numeric_cols if col in data.columns and data[col].notna().sum() > 0]
            
            if available_cols:
                metric_col = available_cols[0]
                chart_title = f'{metric_col.replace("_", " ").title()} Comparison'
        
        if not metric_col or metric_col not in data.columns or 'State' not in data.columns:
            print("[DEBUG] Insufficient data for comparison chart")
            return None, None
        
        # Remove rows with NaN values in the metric column
        clean_data = data.dropna(subset=[metric_col, 'State'])
        
        if clean_data.empty:
            print("[DEBUG] No valid data after cleaning")
            return None, None
        
        # Group by state and calculate mean
        comparison_data = clean_data.groupby('State')[metric_col].mean().reset_index()
        comparison_data = comparison_data.sort_values(metric_col, ascending=False)  # Sort for better visualization
        
        chart_data = {
            "type": "bar",
            "title": chart_title,
            "xAxis": {
                "label": "State",
                "type": "category",
                "categories": list(comparison_data['State'])
            },
            "yAxis": {
                "label": metric_col.replace("_", " ").title(),
                "type": "value"
            },
            "series": [{
                "name": metric_col.replace("_", " ").title(),
                "type": "bar",
                "data": [round(float(val), 2) for val in comparison_data[metric_col]],
                "color": "#007bff"
            }]
        }
        
        print(f"[DEBUG] Comparison BAR chart created with {len(comparison_data)} data points")
        print(f"[DEBUG] Chart data preview: {chart_data}")
        return chart_data, "bar"

    def create_trend_line_chart_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create LINE chart for trend analysis"""
        print(f"[DEBUG] Creating trend LINE chart")
        
        if 'Year' not in data.columns:
            print("[DEBUG] No Year column found")
            return None, None
            
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns and data[col].notna().sum() > 0]
        
        if not available_cols:
            print("[DEBUG] No numeric columns with data found")
            return None, None
        
        chart_data = {
            "type": "line",
            "title": f'{available_cols[0].replace("_", " ").title()} Trend Over Time',
            "xAxis": {
                "label": "Year",
                "type": "category"
            },
            "yAxis": {
                "label": available_cols[0].replace("_", " ").title(),
                "type": "value"
            },
            "series": []
        }
        
        # Group data by state if multiple states present
        if 'State' in data.columns and data['State'].nunique() > 1:
            chart_data["title"] += " by State"
            
            for state in data['State'].dropna().unique():
                state_data = data[data['State'] == state]
                if state_data.empty:
                    continue
                    
                # Group by year and calculate mean, handling NaN values
                trend_data = state_data.groupby('Year')[available_cols[0]].mean().reset_index()
                trend_data = trend_data.dropna()
                
                if not trend_data.empty:
                    chart_data["series"].append({
                        "name": state,
                        "type": "line",
                        "data": [
                            {"x": int(row['Year']), "y": float(row[available_cols[0]])} 
                            for _, row in trend_data.iterrows()
                        ]
                    })
        else:
            # Single series for one state or overall trend
            trend_data = data.groupby('Year')[available_cols[0]].mean().reset_index()
            trend_data = trend_data.dropna()
            
            if not trend_data.empty:
                chart_data["series"].append({
                    "name": available_cols[0].replace("_", " ").title(),
                    "type": "line",
                    "data": [
                        {"x": int(row['Year']), "y": float(row[available_cols[0]])} 
                        for _, row in trend_data.iterrows()
                    ]
                })
        
        print(f"[DEBUG] Trend LINE chart created with {len(chart_data['series'])} series")
        return chart_data, "line"


    def create_trend_chart_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create trend line chart with proper data points"""
        print(f"[DEBUG] Creating trend chart")
        
        if 'Year' not in data.columns:
            print("[DEBUG] No Year column found")
            return None, None
            
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns and data[col].notna().sum() > 0]
        
        if not available_cols:
            print("[DEBUG] No numeric columns with data found")
            return None, None
        
        chart_data = {
            "type": "line",
            "title": f'{available_cols[0].replace("_", " ").title()} Trend Over Time',
            "xAxis": {
                "label": "Year",
                "type": "category"
            },
            "yAxis": {
                "label": available_cols[0].replace("_", " ").title(),
                "type": "value"
            },
            "series": []
        }
        
        # Group data by state if multiple states present
        if 'State' in data.columns and data['State'].nunique() > 1:
            chart_data["title"] += " by State"
            
            for state in data['State'].dropna().unique():
                state_data = data[data['State'] == state]
                if state_data.empty:
                    continue
                    
                # Group by year and calculate mean, handling NaN values
                trend_data = state_data.groupby('Year')[available_cols[0]].mean().reset_index()
                trend_data = trend_data.dropna()  # Remove any rows with NaN
                
                if not trend_data.empty:
                    chart_data["series"].append({
                        "name": state,
                        "type": "line",
                        "data": [
                            {"x": int(row['Year']), "y": float(row[available_cols[0]])} 
                            for _, row in trend_data.iterrows()
                        ]
                    })
        else:
            # Single series for one state or overall trend
            trend_data = data.groupby('Year')[available_cols[0]].mean().reset_index()
            trend_data = trend_data.dropna()  # Remove any rows with NaN
            
            if not trend_data.empty:
                chart_data["series"].append({
                    "name": available_cols[0].replace("_", " ").title(),
                    "type": "line",
                    "data": [
                        {"x": int(row['Year']), "y": float(row[available_cols[0]])} 
                        for _, row in trend_data.iterrows()
                    ]
                })
        
        print(f"[DEBUG] Trend chart created with {len(chart_data['series'])} series")
        return chart_data, "line"

    def create_comparison_chart_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create comparison bar chart including Stage of Extraction"""
        print(f"[DEBUG] Creating comparison chart")
        
        # Check if we should create a stage extraction chart
        if 'State' in data.columns and 'Stage_of_Extraction' in data.columns:
            # Check if the data has stage extraction info
            has_stage_data = data['Stage_of_Extraction'].notna().sum() > 0
            if has_stage_data:
                return self.create_stage_comparison_chart_json(data, "")
        
        # Fall back to numeric comparison
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns and data[col].notna().sum() > 0]
        
        if not available_cols or 'State' not in data.columns:
            print("[DEBUG] Insufficient data for comparison chart")
            return None, None
        
        # Remove rows with NaN values in the metric column
        clean_data = data.dropna(subset=[available_cols[0]])
        comparison_data = clean_data.groupby('State')[available_cols[0]].mean().reset_index()
        
        chart_data = {
            "type": "bar",
            "title": f'{available_cols[0].replace("_", " ").title()} by State',
            "xAxis": {
                "label": "State",
                "type": "category"
            },
            "yAxis": {
                "label": available_cols[0].replace("_", " ").title(),
                "type": "value"
            },
            "series": [{
                "name": available_cols[0].replace("_", " ").title(),
                "type": "bar",
                "data": [
                    {"x": row['State'], "y": float(row[available_cols[0]])} 
                    for _, row in comparison_data.iterrows()
                ]
            }]
        }
        
        print(f"[DEBUG] Comparison chart created with {len(comparison_data)} data points")
        return chart_data, "bar"

    def create_distribution_pie_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create PIE chart for distribution/breakdown queries"""
        print(f"[DEBUG] Creating distribution PIE chart")
        
        # Determine what to show distribution for
        if 'stage' in query and 'Stage_of_Extraction' in data.columns:
            # Distribution of stages
            clean_data = data.dropna(subset=['Stage_of_Extraction'])
            if clean_data.empty:
                return None, None
                
            distribution = clean_data['Stage_of_Extraction'].value_counts()
            
            # Define colors for stages
            stage_colors = {
                'Safe': '#28a745',
                'Semi-Critical': '#ffc107', 
                'Critical': '#fd7e14',
                'Over-Exploited': '#dc3545',
                'Over-Critical': '#dc3545'
            }
            
            chart_data = {
                "type": "pie",
                "title": "Distribution of Stage of Extraction",
                "series": [{
                    "name": "Stage Distribution",
                    "type": "pie",
                    "data": [
                        {
                            "name": stage,
                            "value": int(count),
                            "color": stage_colors.get(stage, '#007bff')
                        }
                        for stage, count in distribution.items()
                    ]
                }]
            }
            
        elif 'state' in query and 'State' in data.columns:
            # Distribution by states
            distribution = data['State'].value_counts().head(10)  # Top 10 states
            
            chart_data = {
                "type": "pie", 
                "title": "Distribution by State",
                "series": [{
                    "name": "State Distribution",
                    "type": "pie",
                    "data": [
                        {
                            "name": state,
                            "value": int(count)
                        }
                        for state, count in distribution.items()
                    ]
                }]
            }
            
        else:
            print("[DEBUG] No suitable data for pie chart distribution")
            return None, None
        
        print(f"[DEBUG] Distribution PIE chart created")
        return chart_data, "pie"

    def create_state_average_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create chart for state average queries"""
        print(f"[DEBUG] Creating state average chart")
        
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        
        # Determine which metric to show based on query
        metric_col = 'Groundwater_Level_m'  # default
        if 'recharge' in query:
            metric_col = 'Recharge_mm' if 'Recharge_mm' in data.columns else metric_col
        elif 'extraction' in query:
            metric_col = 'Extraction_mm' if 'Extraction_mm' in data.columns else metric_col
        
        if metric_col not in data.columns or data[metric_col].notna().sum() == 0:
            print(f"[DEBUG] No data available for metric: {metric_col}")
            return None, None
        
        # If we have year data, show trend over years
        if 'Year' in data.columns and data['Year'].nunique() > 1:
            clean_data = data.dropna(subset=[metric_col, 'Year'])
            yearly_avg = clean_data.groupby('Year')[metric_col].mean().reset_index()
            
            chart_data = {
                "type": "line",
                "title": f'{metric_col.replace("_", " ").title()} Average Over Years',
                "xAxis": {
                    "label": "Year",
                    "type": "category"
                },
                "yAxis": {
                    "label": metric_col.replace("_", " ").title(),
                    "type": "value"
                },
                "series": [{
                    "name": metric_col.replace("_", " ").title(),
                    "type": "line",
                    "data": [
                        {"x": int(row['Year']), "y": float(row[metric_col])} 
                        for _, row in yearly_avg.iterrows()
                    ]
                }]
            }
            
            print(f"[DEBUG] State average chart created with {len(yearly_avg)} data points")
            return chart_data, "line"
        
        return None, None

    def clean_response(self, response: str) -> str:
        """Clean up agent response to remove action/thought artifacts and dataset mentions"""
        import re
        
        # Remove Action and Thought sections
        response = re.sub(r'Action:\s*python_repl_ast.*?(?=Thought:|Final Answer:|$)', '', response, flags=re.DOTALL)
        response = re.sub(r'Thought:.*?(?=Final Answer:|Thought:|$)', '', response, flags=re.DOTALL)
        
        # Extract Final Answer if present
        final_answer_match = re.search(r'Final Answer:\s*(.*?)(?=Action:|Thought:|$)', response, re.DOTALL)
        if final_answer_match:
            response = final_answer_match.group(1).strip()
        
        # Remove mentions of dataset details
        response = re.sub(r'There are \d+ rows?.*?(?=\.|\n|$)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'The dataframe.*?(?=\.|\n|$)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Based on the data.*?(?=\.|\n|$)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'In the dataset.*?(?=\.|\n|$)', '', response, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()
        
        return response
    
    def is_greeting_query(self, query: str) -> bool:
        """Check if query is a simple greeting"""
        query_lower = query.lower().strip()
        simple_greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
                           'how are you', 'how you doing', 'what\'s up', 'greetings']
        return query_lower in simple_greetings 
    
    def handle_greeting_query(self, query: str) -> str:
        """Handle greeting queries without using the agent"""
        return "Hello! I'm your groundwater data assistant. You can ask me about groundwater levels, recharge rates, extraction data, and trends across different states and years."

    def extract_relevant_data(self, query: str) -> Optional[pd.DataFrame]:
        """Extract relevant data subset based on query keywords with improved matching"""
        query_lower = query.lower()
        
        try:
            # Find mentioned states using improved extraction
            mentioned_states = self.extract_states_from_query(query)
            
            # Find mentioned year
            mentioned_year = self.extract_year_from_query(query)
            
            print(f"[DEBUG] Extracting data - States: {mentioned_states}, Year: {mentioned_year}")
            
            # Filter data
            filtered_df = self.df.copy()
            
            if mentioned_states:
                filtered_df = filtered_df[filtered_df['State'].isin(mentioned_states)]
            
            if mentioned_year:
                filtered_df = filtered_df[filtered_df['Year'] == mentioned_year]
            
            # For trend queries, get more years of data
            if 'trend' in query_lower and not mentioned_year:
                # Get last 5 years of data or all available years
                available_years = sorted(filtered_df['Year'].dropna().unique())
                if len(available_years) > 5:
                    recent_years = available_years[-5:]
                    filtered_df = filtered_df[filtered_df['Year'].isin(recent_years)]
            
            print(f"[DEBUG] Extracted data shape: {filtered_df.shape}")
            if not filtered_df.empty:
                print(f"[DEBUG] Sample of extracted data:")
                print(filtered_df[['State', 'Year', 'Stage_of_Extraction']].head())
            
            return filtered_df if not filtered_df.empty else None
            
        except Exception as e:
            print(f"[DEBUG] Error extracting relevant data: {str(e)}")
            return None

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset"""
        return {
            "total_records": len(self.df),
            "states_count": self.df['State'].nunique(),
            "years_range": (int(self.df['Year'].min()), int(self.df['Year'].max())),
            "available_years": sorted(self.df['Year'].unique().tolist()),
            "available_states": sorted(self.df['State'].unique().tolist())
        }

    

# Initialize assistant globally
assistant = None

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on startup"""
    global assistant
    csv_path = "data/ground_water_data.csv"  # Adjust path as needed
    assistant = GroundwaterAssistant(csv_path)
    print("Groundwater Assistant initialized")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "INGRES Groundwater Data Assistant API", "version": "1.0.0"}

@app.head("/")
async def head_root():
    """Handle HEAD requests for health checks"""
    from fastapi import Response
    return Response(status_code=200)



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "data_loaded": not assistant.df.empty if assistant else False}

@app.get("/summary", response_model=DataSummary)
async def get_data_summary():
    """Get data summary"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    summary = assistant.get_data_summary()
    return DataSummary(**summary)

@app.get("/states")
async def get_states():
    """Get list of available states"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    return {"states": sorted(assistant.df['State'].unique().tolist())}

@app.get("/years")
async def get_years():
    """Get list of available years"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    return {"years": sorted(assistant.df['Year'].unique().tolist())}

@app.get("/columns")
async def get_columns():
    """Get list of available columns"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    return {"columns": assistant.df.columns.tolist()}

@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """Process user query"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = assistant.query_data(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)