from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import dotenv
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import json
import uvicorn
import numpy as np
import re
import math

# LangChain imports
from langchain_experimental.agents import create_pandas_dataframe_agent
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
dotenv.load_dotenv()

app = FastAPI(
    title="INGRES Enhanced Groundwater API",
    description="AI-powered groundwater data analysis API with location support",
    version="2.0.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change * to your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Pydantic models
class QueryRequest(BaseModel):
    question: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: Optional[float] = 50.0  # Default 50km radius

# class DataSummary(BaseModel):
#     total_records: int
#     states_count: int
#     districts_count: int
#     cities_count: int
#     years_range: tuple
#     available_years: List[int]
#     available_states: List[str]
#     available_districts: List[str]
#     available_cities: List[str]

class QueryResponse(BaseModel):
    response: str 
    data: Optional[List[Dict]] = None
    chart: Optional[Dict] = None
    chart_type: Optional[str] = None
    location_info: Optional[Dict] = None

class SimpleResponse(BaseModel):
    response: str

class PredictionRequest(BaseModel):
    coordinates: Optional[Dict[str, float]] = None
    location: Optional[Dict[str, Any]] = None

class GroundwaterAssistant:
    """Enhanced Groundwater Assistant with location support"""
    
    def __init__(self, csv_path: str):
        """Initialize the assistant with CSV data and PandasAI agent"""
        self.df = self.load_data(csv_path)
        self.agent = None
        self.langchain_agent = None
        self.setup_pandas_ai_agent()
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess the enhanced CSV data"""
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(csv_path)
        
        print(f"[DEBUG] Raw data loaded: {len(df)} rows")
        print(f"[DEBUG] Columns: {list(df.columns)}")
        if 'Category' in df.columns:
            try:
                print(f"[DEBUG] First few Category values: {df['Category'].head(10).tolist()}")
                print(f"[DEBUG] Unique Categories: {df['Category'].unique()}")
                print(f"[DEBUG] Category value counts:\n{df['Category'].value_counts()}")
            except Exception as e:
                print(f"[DEBUG] Category debug error: {str(e)}")
        print(f"[DEBUG] Data types:\n{df.dtypes}")
        
        # Basic data cleaning - handle null values properly
        df = df.dropna(subset=['State'], how='all')
        
        # Convert numeric columns
        numeric_columns = [
            'Year', 'Population', 'Area (in km^2)', 'Latitude', 'Longitude',
            'Annual_Rainfall_mm', 'Annual_Recharge_MCM', 'Annual_Extractable_MCM',
            'Annual_Extraction_MCM', 'Stage_of_Extraction_pct', 'Is_Saline',
            'Groundwater_Level_m'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean text columns
        text_columns = ['State', 'District', 'City', 'Category']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Debug after cleaning
        if 'Category' in df.columns:
            print(f"[DEBUG] After cleaning - Category value counts:\n{df['Category'].value_counts()}")
        print(f"[DEBUG] Sample rows:")
        cols_for_sample = [c for c in ['State', 'District', 'Category', 'Population'] if c in df.columns]
        try:
            print(df[cols_for_sample].head())
        except Exception:
            print(df.head())
        
        return df
    
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r

    def find_nearest_locations(self, latitude: float, longitude: float, radius_km: float = 50.0) -> pd.DataFrame:
        """Find locations within radius, sorted by distance"""
        if 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns:
            return pd.DataFrame()
        
        # Remove rows with missing coordinates
        valid_coords = self.df.dropna(subset=['Latitude', 'Longitude'])
        
        if valid_coords.empty:
            return pd.DataFrame()
        
        # Calculate distances
        distances = []
        for _, row in valid_coords.iterrows():
            dist = self.calculate_distance(latitude, longitude, row['Latitude'], row['Longitude'])
            distances.append(dist)
        
        valid_coords = valid_coords.copy()
        valid_coords['Distance_km'] = distances
        
        # Filter by radius and sort by distance
        nearby = valid_coords[valid_coords['Distance_km'] <= radius_km]
        nearby = nearby.sort_values('Distance_km')
        
        print(f"[DEBUG] Found {len(nearby)} locations within {radius_km}km")
        
        return nearby

    def extract_location_keywords(self, query: str) -> Dict[str, Any]:
        """Extract location-related keywords and intent from query"""
        query_lower = query.lower()
        
        location_intent = {
            'is_current_location': any(phrase in query_lower for phrase in [
                'current location', 'my location', 'here', 'near me', 'nearby', 'around me',
                'at my location', 'where i am', 'in my area', 'locally', 'in this area'
            ]),
            'mentioned_state': None,
            'mentioned_district': None,
            'mentioned_city': None,
            'requires_coordinates': False
        }
        
        # Extract mentioned locations
        if 'State' in self.df.columns:
            states = self.df['State'].dropna().unique()
            for state in states:
                if state.lower() in query_lower:
                    location_intent['mentioned_state'] = state
                    break
        
        if 'District' in self.df.columns:
            districts = self.df['District'].dropna().unique()
            for district in districts:
                if district.lower() in query_lower:
                    location_intent['mentioned_district'] = district
                    break
        
        if 'City' in self.df.columns:
            cities = self.df['City'].dropna().unique()
            for city in cities:
                if city.lower() in query_lower:
                    location_intent['mentioned_city'] = city
                    break
        
        location_intent['requires_coordinates'] = location_intent['is_current_location']
        
        return location_intent

    def setup_pandas_ai_agent(self):
        """Setup PandasAI agent with Groq"""
        try:
            load_dotenv()
            groq_key = os.getenv("GROQ_API_KEY")
            
            if not groq_key or not groq_key.startswith("gsk_"):
                print("[DEBUG] GROQ API key not found. Please set GROQ_API_KEY in environment")
                print("[DEBUG] Continuing without AI agent - will use direct data analysis")
                self.langchain_agent = None
                return
            
            llm_groq = ChatGroq(
                groq_api_key=groq_key,
                model="meta-llama/llama-3.2-70b-versatile",
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

    def query_data(self, user_question: str, latitude: Optional[float] = None, longitude: Optional[float] = None) -> Dict[str, Any]:
        """Process enhanced user query with location support"""
        # EXPLICIT VISUALIZATION FORCING
        query_lower = user_question.lower().strip()
        
        if query_lower == "show category distribution":
            print("[DEBUG] FORCING category distribution visualization")
            chart_data, chart_type = self.create_category_distribution_pie_json(self.df)
            response_text = self.generate_viz_response(user_question, self.df, None)
            return {
                "response": response_text,
                "data": None,
                "chart": chart_data,
                "chart_type": chart_type,
                "location_info": None
            }
        
        if query_lower == "show population by state":
            print("[DEBUG] FORCING population by state visualization")
            chart_data, chart_type = self.create_population_chart_json(self.df)
            response_text = self.generate_viz_response(user_question, self.df, None)
            return {
                "response": response_text,
                "data": None,
                "chart": chart_data,
                "chart_type": chart_type,
                "location_info": None
            }

        # Enhance the input query first (temporarily bypassing enhance_input)
        enhanced_question = user_question
        print(f"[DEBUG] Original query: '{user_question}'")
        print(f"[DEBUG] Enhanced query: '{enhanced_question}'")
        
        # Check if visualization is required FIRST (use raw user question to catch phrases)
        requires_viz = self.requires_visualization(user_question)
        print(f"[DEBUG] Requires visualization: {requires_viz}")
        
        try:
            # Extract location intent
            location_intent = self.extract_location_keywords(enhanced_question)
            location_info = None
            filtered_data = self.df.copy()
            
            # Extract explicit states for potential filtering in comparisons
            extracted_states = self.extract_states_from_query(enhanced_question)
            print(f"[DEBUG] Extracted states: {extracted_states}")
            # Only apply state filtering for explicit comparison-style queries
            comparison_keywords = ['compare', 'comparison', 'vs', 'versus', 'between']
            query_lower_for_filter = user_question.lower()
            if extracted_states and any(k in query_lower_for_filter for k in comparison_keywords):
                filtered_data = self.df[self.df['State'].isin(extracted_states)]
                print(f"[DEBUG] Filtered data for comparison states {extracted_states}: {len(filtered_data)} rows")
            
            # Handle location-based queries (keep existing logic)
            if location_intent['is_current_location']:
                if latitude is not None and longitude is not None:
                    nearby_locations = self.find_nearest_locations(latitude, longitude, 50.0)
                    if not nearby_locations.empty:
                        filtered_data = nearby_locations
                        closest = nearby_locations.iloc[0]
                        location_info = {
                            'user_location': {'latitude': latitude, 'longitude': longitude},
                            'closest_location': {
                                'state': closest.get('State'),
                                'district': closest.get('District'),
                                'city': closest.get('City'),
                                'distance_km': round(closest['Distance_km'], 2)
                            },
                            'locations_found': len(nearby_locations)
                        }
                    else:
                        return {
                            "response": f"No groundwater data found within 50km of your location.",
                            "data": None, "chart": None, "chart_type": None, "location_info": None
                        }
                else:
                    return {
                        "response": "This query requires your location, but location data wasn't provided.",
                        "data": None, "chart": None, "chart_type": None,
                        "location_info": {"error": "location_required", "message": "Location access needed"}
                    }
            
            # Handle specific location mentions
            elif location_intent['mentioned_state'] or location_intent['mentioned_district'] or location_intent['mentioned_city']:
                if location_intent['mentioned_city']:
                    filtered_data = filtered_data[filtered_data['City'] == location_intent['mentioned_city']]
                elif location_intent['mentioned_district']:
                    filtered_data = filtered_data[filtered_data['District'] == location_intent['mentioned_district']]
                elif location_intent['mentioned_state']:
                    filtered_data = filtered_data[filtered_data['State'] == location_intent['mentioned_state']]
            

            # Handle greeting queries
            if self.is_greeting_query(enhanced_question):
                return {
                    "response": self.handle_greeting_query(enhanced_question),
                    "data": None, "chart": None, "chart_type": None, "location_info": location_info
                }
            
            # Handle non-groundwater queries
            if not self.is_groundwater_query(enhanced_question):
                return {
                    "response": self.handle_general_query(enhanced_question),
                    "data": None, "chart": None, "chart_type": None, "location_info": location_info
                }
            
            # FOR VISUALIZATION QUERIES - Handle directly without AI agent
            if requires_viz:
                print("[DEBUG] Processing as visualization query")
                # For comparisons use state-filtered data; for general visualizations use full dataset
                is_comparison = any(k in user_question.lower() for k in ['compare', 'comparison', 'vs', 'versus', 'between'])
                data_for_viz = filtered_data if is_comparison else self.df
                # Create visualization first
                chart_data, chart_type = self.create_enhanced_visualization_json(user_question, data_for_viz)
                # Generate response text
                response_text = self.generate_viz_response(user_question, data_for_viz, location_info)
                return {
                    "response": response_text,
                    "data": None,
                    "chart": chart_data,
                    "chart_type": chart_type,
                    "location_info": location_info
                }
            
            # For non-visualization queries, try direct handling first
            direct_result = self.handle_enhanced_direct_query(enhanced_question, filtered_data, location_info)
            if direct_result:
                return {
                    "response": direct_result,
                    "data": None, "chart": None, "chart_type": None, "location_info": location_info
                }
            
            # Use AI agent as fallback
            if self.langchain_agent:
                # Keep existing AI agent prompt & logic
                GROUNDWATER_PROMPT = """
You are the Groundwater Dataset Assistant.

- If the user's question is about groundwater levels, wells, recharge, extraction, rainfall, aquifers, water table, or related metrics:
  • Answer ONLY using the provided dataset.  
  • Never invent values.  
  • If the dataset doesn't contain the answer, reply:  
    "I don't have enough data in the Dataset to answer that precisely."  
  • Include references (state, district, year, record count, etc.).  

- If the question is NOT about groundwater, politely hand off:  
  "This assistant answers groundwater dataset questions only. For general queries, I can respond as a general assistant."  
"""
                enhanced_for_agent = f"""
                {GROUNDWATER_PROMPT}
                
                User Question: {enhanced_question}
                
                Dataset context: The dataset contains groundwater information with columns:
                State, District, City, Population, Area (in km^2), Latitude, Longitude, Year,
                Annual_Rainfall_mm, Annual_Recharge_MCM, Annual_Extractable_MCM, Annual_Extraction_MCM,
                Stage_of_Extraction_pct, Category (Safe/Semi-Critical/Critical/Over-Exploited), Is_Saline, Groundwater_Level_m
                
                Instructions:
                - Provide clear, conversational responses
                - For numerical results, include metric names and values
                - Handle missing data gracefully
                - Don't mention dataset details or show code
                """
                if location_intent['is_current_location'] and not filtered_data.empty:
                    temp_agent = create_pandas_dataframe_agent(
                        ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="meta-llama/llama-3.2-70b-versatile", temperature=0),
                        filtered_data,
                        verbose=False,
                        allow_dangerous_code=True
                    )
                    response = temp_agent.run(enhanced_for_agent)
                else:
                    response = self.langchain_agent.run(enhanced_for_agent)
                response = self.clean_response(response)
            else:
                if filtered_data is not None and not filtered_data.empty:
                    response = f"I found {len(filtered_data)} records matching your query."
                else:
                    response = "I don't have enough data to answer that precisely."
            
            return {
                "response": str(response),
                "data": None, "chart": None, "chart_type": None, "location_info": location_info
            }

        except Exception as e:
            print(f"[DEBUG] Error processing query: {str(e)}")
            return {
                "response": "I'm having trouble processing that question. Could you please rephrase it?",
                "data": None, "chart": None, "chart_type": None, "location_info": None
            }
        print(f"[DEBUG] Original query: '{user_question}'")
        print(f"[DEBUG] Requires visualization: {self.requires_visualization(user_question)}")
        print(f"[DEBUG] Is groundwater query: {self.is_groundwater_query(user_question)}")

    def handle_enhanced_direct_query(self, query: str, data: pd.DataFrame, location_info: Optional[Dict]) -> Optional[str]:
        """Handle enhanced direct queries with new dataset fields"""
        try:
            query_lower = query.lower()
            
            if data.empty:
                return "No data found for the specified criteria."
            
            location_prefix = ""
            if location_info and location_info.get('closest_location'):
                closest = location_info['closest_location']
                location_prefix = f"Near your location ({closest['city']}, {closest['district']}, {closest['state']} - {closest['distance_km']}km away): "
            
            # Population-based queries
            if 'population' in query_lower:
                if 'Population' in data.columns:
                    avg_pop = data['Population'].mean()
                    total_pop = data['Population'].sum()
                    if 'average' in query_lower:
                        return f"{location_prefix}The average population is {avg_pop:,.0f} people."
                    elif 'total' in query_lower:
                        return f"{location_prefix}The total population is {total_pop:,.0f} people."
            
            # Area-based queries
            if 'area' in query_lower:
                if 'Area (in km^2)' in data.columns:
                    total_area = data['Area (in km^2)'].sum()
                    avg_area = data['Area (in km^2)'].mean()
                    if 'total' in query_lower:
                        return f"{location_prefix}The total area is {total_area:,.2f} km²."
                    elif 'average' in query_lower:
                        return f"{location_prefix}The average area is {avg_area:,.2f} km²."
            
            # Rainfall queries
            if 'rainfall' in query_lower:
                if 'Annual_Rainfall_mm' in data.columns:
                    avg_rainfall = data['Annual_Rainfall_mm'].mean()
                    return f"{location_prefix}The average annual rainfall is {avg_rainfall:.1f} mm."
            
            # Recharge queries
            if 'recharge' in query_lower:
                if 'Annual_Recharge_MCM' in data.columns:
                    total_recharge = data['Annual_Recharge_MCM'].sum()
                    avg_recharge = data['Annual_Recharge_MCM'].mean()
                    if 'total' in query_lower:
                        return f"{location_prefix}The total annual recharge is {total_recharge:.2f} MCM."
                    else:
                        return f"{location_prefix}The average annual recharge is {avg_recharge:.2f} MCM."
            
            # Extraction queries
            if 'extraction' in query_lower and 'stage' not in query_lower:
                if 'Annual_Extraction_MCM' in data.columns:
                    total_extraction = data['Annual_Extraction_MCM'].sum()
                    avg_extraction = data['Annual_Extraction_MCM'].mean()
                    if 'total' in query_lower:
                        return f"{location_prefix}The total annual extraction is {total_extraction:.2f} MCM."
                    else:
                        return f"{location_prefix}The average annual extraction is {avg_extraction:.2f} MCM."
            
            # Stage of extraction percentage queries
            if 'stage' in query_lower and 'percentage' in query_lower:
                if 'Stage_of_Extraction_pct' in data.columns:
                    avg_stage = data['Stage_of_Extraction_pct'].mean()
                    return f"{location_prefix}The average stage of extraction is {avg_stage:.1f}%."
            
            # Category-based queries
            if 'category' in query_lower or ('safe' in query_lower or 'critical' in query_lower or 'exploited' in query_lower):
                if 'Category' in data.columns:
                    categories = data['Category'].value_counts()
                    if len(categories) > 0:
                        result = f"{location_prefix}Category distribution: "
                        result += ", ".join([f"{cat}: {count}" for cat, count in categories.items()])
                        return result
            
            # Saline water queries
            if 'saline' in query_lower:
                if 'Is_Saline' in data.columns:
                    saline_count = data['Is_Saline'].sum()
                    total_count = len(data)
                    saline_pct = (saline_count / total_count) * 100 if total_count > 0 else 0
                    return f"{location_prefix}Saline groundwater: {saline_count} out of {total_count} locations ({saline_pct:.1f}%)."
            
            # Groundwater level queries (handle typos)
            query_lower_fixed = query_lower.replace('grndwater', 'groundwater')
            if 'groundwater' in query_lower_fixed and 'level' in query_lower_fixed:
                if 'Groundwater_Level_m' in data.columns:
                    avg_level = data['Groundwater_Level_m'].mean()
                    return f"{location_prefix}The average groundwater level is {avg_level:.1f} meters."
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Error in handle_enhanced_direct_query: {str(e)}")
            return None

    def create_enhanced_visualization_json(self, query: str, data: Optional[pd.DataFrame]) -> tuple[Optional[Dict], Optional[str]]:
        """Create enhanced visualizations for the new dataset"""
        print(f"[DEBUG] Creating enhanced visualization for query: {query}")
        
        if data is None or data.empty:
            print("[DEBUG] No data available for visualization")
            return None, None
            
        query_lower = query.lower()
        
        try:
            print(f"[DEBUG] Query contains 'rainfall': {'rainfall' in query_lower}")
            print(f"[DEBUG] Query contains 'compare': {'compare' in query_lower}")
            # Groundwater level specific comparison
            if ('groundwater' in query_lower and 'level' in query_lower) and ('compare' in query_lower or 'comparison' in query_lower or 'vs' in query_lower or 'versus' in query_lower or 'between' in query_lower):
                print("[DEBUG] Creating groundwater level comparison")
                return self.create_groundwater_level_comparison_json(data)
            # Category distribution - explicit + keyword
            if ('category' in query_lower and 'distribution' in query_lower) or 'show category distribution' in query_lower:
                print("[DEBUG] Creating category distribution pie chart")
                return self.create_category_distribution_pie_json(data)
            
            # Population by state - explicit + keyword
            elif ('population' in query_lower and 'state' in query_lower) or 'show population by state' in query_lower:
                print("[DEBUG] Creating population by state bar chart")
                return self.create_population_chart_json(data)
            
            # Rainfall comparison by state - multi-state specific then general
            elif 'rainfall' in query_lower and ('compare' in query_lower or 'comparison' in query_lower or 'by state' in query_lower or 'by city' in query_lower or 'by district' in query_lower):
                extracted_states = self.extract_states_from_query(query)
                if extracted_states and len(extracted_states) >= 2:
                    print(f"[DEBUG] Creating multi-state rainfall comparison for: {extracted_states}")
                    return self.create_multi_state_rainfall_comparison_json(data, extracted_states)
                else:
                    print("[DEBUG] Creating general rainfall comparison")
                    return self.create_rainfall_comparison_json(data, query)

            # Rainfall trends - LINE CHART
            elif 'rainfall' in query_lower and 'trend' in query_lower:
                return self.create_rainfall_trend_json(data)
            
            # Extraction vs Recharge comparison - BAR CHART (require both keywords)
            elif ('extraction' in query_lower and 'recharge' in query_lower):
                print("[DEBUG] Matched condition: extraction vs recharge comparison")
                return self.create_extraction_recharge_comparison_json(data)
            
            # Saline water distribution - PIE CHART
            elif 'saline' in query_lower and 'distribution' in query_lower:
                return self.create_saline_distribution_json(data)
            
            # District/City comparison - BAR CHART
            elif ('district' in query_lower or 'city' in query_lower) and 'compare' in query_lower:
                return self.create_location_comparison_json(data, query_lower)
            
            # General location-based trends
            elif any(word in query_lower for word in ['trend', 'over time', 'yearly']):
                return self.create_temporal_trend_json(data, query_lower)
            
            # Fallback to original visualization logic
            return self.create_visualization_json(query, data)
                        
        except Exception as e:
            print(f"[DEBUG] Enhanced chart creation error: {str(e)}")
            return None, None

    def create_groundwater_level_comparison_json(self, data: pd.DataFrame) -> tuple[Optional[Dict], Optional[str]]:
        """Create BAR chart specifically for groundwater level comparison"""
        print(f"[DEBUG] create_groundwater_level_comparison_json called with {len(data)} rows")
        required_cols = ['Groundwater_Level_m', 'State']
        if not all(col in data.columns for col in required_cols):
            print(f"[DEBUG] Missing required columns. Available: {list(data.columns)}")
            return None, None
        clean_data = data.dropna(subset=required_cols)
        if clean_data.empty:
            print("[DEBUG] No data after cleaning")
            return None, None
        state_levels = clean_data.groupby('State')['Groundwater_Level_m'].mean().reset_index()
        state_levels = state_levels.sort_values('Groundwater_Level_m', ascending=False)
        print(f"[DEBUG] Groundwater levels by state:")
        print(state_levels)
        chart_data = {
            "type": "bar",
            "title": "Groundwater Level Comparison by State",
            "xAxis": {
                "label": "State",
                "type": "category",
                "categories": state_levels['State'].tolist()
            },
            "yAxis": {
                "label": "Average Groundwater Level (m)",
                "type": "value"
            },
            "series": [{
                "name": "Groundwater Level",
                "type": "bar",
                "data": [round(float(level), 2) for level in state_levels['Groundwater_Level_m']],
                "color": "#007bff"
            }]
        }
        print(f"[DEBUG] Groundwater level comparison chart created with {len(state_levels)} states")
        return chart_data, "bar"

    def create_category_distribution_pie_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create PIE chart for Category distribution"""
        print(f"[DEBUG] create_category_distribution_pie_json called with {len(data)} rows")
        
        if 'Category' not in data.columns:
            print("[DEBUG] No Category column found")
            return None, None
        
        # Clean and filter data
        clean_data = data.dropna(subset=['Category'])
        print(f"[DEBUG] After dropna: {len(clean_data)} rows")
        
        if clean_data.empty:
            print("[DEBUG] No data after cleaning")
            return None, None
        
        # Get distribution
        distribution = clean_data['Category'].value_counts()
        print(f"[DEBUG] Category distribution: {distribution.to_dict()}")
        
        if distribution.empty:
            print("[DEBUG] Empty distribution")
            return None, None
        
        # Color mapping for categories
        category_colors = {
            'Safe': '#28a745',
            'Semi-Critical': '#ffc107', 
            'Critical': '#fd7e14', 
            'Over-Exploited': '#dc3545',
            'Over-Critical': '#dc3545'
        }
        
        # Create chart data
        chart_data = {
            "type": "pie",
            "title": "Groundwater Category Distribution",
            "series": [{
                "name": "Category Distribution",
                "type": "pie",
                "data": []
            }]
        }
        
        # Build data array
        for category, count in distribution.items():
            color = category_colors.get(str(category), '#007bff')
            chart_data["series"][0]["data"].append({
                "name": str(category),
                "value": int(count),
                "color": color
            })
        
        print(f"[DEBUG] Chart data created: {chart_data}")
        return chart_data, "pie"

    def create_population_chart_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create chart for population data"""
        print(f"[DEBUG] create_population_chart_json called with {len(data)} rows")
        
        if 'Population' not in data.columns or 'State' not in data.columns:
            print(f"[DEBUG] Missing columns. Available: {list(data.columns)}")
            return None, None
        
        # Clean data
        clean_data = data.dropna(subset=['Population', 'State'])
        print(f"[DEBUG] Clean data: {len(clean_data)} rows")
        
        if clean_data.empty:
            return None, None
        
        # Group by state and sum population
        try:
            state_pop = clean_data.groupby('State')['Population'].sum().reset_index()
            state_pop = state_pop.sort_values('Population', ascending=False).head(10)
            
            print(f"[DEBUG] State population data:")
            print(state_pop)
            
            chart_data = {
                "type": "bar",
                "title": "Population by State (Top 10)",
                "xAxis": {
                    "label": "State",
                    "type": "category",
                    "categories": state_pop['State'].tolist()
                },
                "yAxis": {
                    "label": "Population",
                    "type": "value"
                },
                "series": [{
                    "name": "Population",
                    "type": "bar",
                    "data": state_pop['Population'].astype(int).tolist(),
                    "color": "#007bff"
                }]
            }
            
            print(f"[DEBUG] Population chart created: {chart_data}")
            return chart_data, "bar"
            
        except Exception as e:
            print(f"[DEBUG] Error creating population chart: {str(e)}")
            return None, None

    def create_rainfall_trend_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create LINE chart for rainfall trends"""
        if 'Annual_Rainfall_mm' not in data.columns or 'Year' not in data.columns:
            return None, None
        
        clean_data = data.dropna(subset=['Annual_Rainfall_mm', 'Year'])
        if clean_data.empty:
            return None, None
        
        yearly_rainfall = clean_data.groupby('Year')['Annual_Rainfall_mm'].mean().reset_index()
        
        chart_data = {
            "type": "line",
            "title": "Annual Rainfall Trend",
            "xAxis": {
                "label": "Year",
                "type": "category"
            },
            "yAxis": {
                "label": "Annual Rainfall (mm)",
                "type": "value"
            },
            "series": [{
                "name": "Average Rainfall",
                "type": "line",
                "data": [
                    {"x": int(row['Year']), "y": float(row['Annual_Rainfall_mm'])} 
                    for _, row in yearly_rainfall.iterrows()
                ]
            }]
        }
        
        return chart_data, "line"

    def create_rainfall_comparison_json(self, data: pd.DataFrame, query: str) -> tuple[Optional[Dict], Optional[str]]:
        """Create BAR chart comparing average rainfall by state"""
        print(f"[DEBUG] create_rainfall_comparison_json called with {len(data)} rows")
        try:
            states_list = data['State'].unique().tolist() if 'State' in data.columns else []
        except Exception:
            states_list = []
        print(f"[DEBUG] Unique states count: {len(states_list)}")
        if 'State' in data.columns and 'Annual_Rainfall_mm' in data.columns:
            print("[DEBUG] Sample rows for rainfall comparison:")
            print(data[['State', 'Annual_Rainfall_mm']].head())
        query_lower = query.lower()
        # Decide grouping level based on query intent
        if 'city' in query_lower and 'City' in data.columns:
            group_col = 'City'
        elif 'district' in query_lower and 'District' in data.columns:
            group_col = 'District'
        else:
            group_col = 'State'
        print(f"[DEBUG] Rainfall comparison grouping by: {group_col}")
        required_cols = ['Annual_Rainfall_mm', group_col]
        if not all(col in data.columns for col in required_cols):
            print(f"[DEBUG] Missing required columns for rainfall comparison. Needed: {required_cols} Available: {list(data.columns)}")
            return None, None
        clean_data = data.dropna(subset=required_cols)
        if clean_data.empty:
            print("[DEBUG] No data after cleaning for rainfall comparison")
            return None, None
        rainfall_by_location = clean_data.groupby(group_col)['Annual_Rainfall_mm'].mean().reset_index()
        rainfall_by_location = rainfall_by_location.sort_values('Annual_Rainfall_mm', ascending=False).head(10)
        chart_data = {
            "type": "bar",
            "title": f"Average Annual Rainfall by {group_col}",
            "xAxis": {
                "label": group_col,
                "type": "category",
                "categories": rainfall_by_location[group_col].tolist()
            },
            "yAxis": {
                "label": "Average Rainfall (mm)",
                "type": "value"
            },
            "series": [{
                "name": "Average Rainfall",
                "type": "bar",
                "data": [float(v) for v in rainfall_by_location['Annual_Rainfall_mm']],
                "color": "#17a2b8"
            }]
        }
        print(f"[DEBUG] Rainfall comparison chart created with {len(rainfall_by_location)} groups by {group_col}")
        return chart_data, "bar"

    def create_extraction_recharge_comparison_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create BAR chart comparing extraction and recharge"""
        required_cols = ['Annual_Extraction_MCM', 'Annual_Recharge_MCM', 'State']
        if not all(col in data.columns for col in required_cols):
            return None, None
        
        clean_data = data.dropna(subset=required_cols)
        if clean_data.empty:
            return None, None
        
        # Group by state
        comparison = clean_data.groupby('State')[['Annual_Extraction_MCM', 'Annual_Recharge_MCM']].mean().reset_index()
        comparison = comparison.head(10)  # Top 10 states
        
        chart_data = {
            "type": "bar",
            "title": "Annual Extraction vs Recharge by State",
            "xAxis": {
                "label": "State",
                "type": "category",
                "categories": list(comparison['State'])
            },
            "yAxis": {
                "label": "Volume (MCM)",
                "type": "value"
            },
            "series": [
                {
                    "name": "Extraction",
                    "type": "bar",
                    "data": [float(val) for val in comparison['Annual_Extraction_MCM']],
                    "color": "#dc3545"
                },
                {
                    "name": "Recharge", 
                    "type": "bar",
                    "data": [float(val) for val in comparison['Annual_Recharge_MCM']],
                    "color": "#28a745"
                }
            ]
        }
        
        return chart_data, "bar"

    def create_saline_distribution_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create PIE chart for saline water distribution"""
        if 'Is_Saline' not in data.columns:
            return None, None
        
        clean_data = data.dropna(subset=['Is_Saline'])
        if clean_data.empty:
            return None, None
        
        saline_count = clean_data['Is_Saline'].sum()
        non_saline_count = len(clean_data) - saline_count
        
        chart_data = {
            "type": "pie",
            "title": "Saline vs Non-Saline Groundwater",
            "series": [{
                "name": "Water Type",
                "type": "pie", 
                "data": [
                    {"name": "Non-Saline", "value": int(non_saline_count), "color": "#28a745"},
                    {"name": "Saline", "value": int(saline_count), "color": "#dc3545"}
                ]
            }]
        }
        
        return chart_data, "pie"

    def create_location_comparison_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create comparison chart for districts or cities"""
        if 'district' in query and 'District' in data.columns:
            group_col = 'District'
            title_suffix = "Districts"
        elif 'city' in query and 'City' in data.columns:
            group_col = 'City'
            title_suffix = "Cities"
        else:
            return None, None
        
        # Determine metric to compare
        metric_col = 'Groundwater_Level_m'  # default
        if 'population' in query and 'Population' in data.columns:
            metric_col = 'Population'
        elif 'rainfall' in query and 'Annual_Rainfall_mm' in data.columns:
            metric_col = 'Annual_Rainfall_mm'
        elif 'recharge' in query and 'Annual_Recharge_MCM' in data.columns:
            metric_col = 'Annual_Recharge_MCM'
        
        clean_data = data.dropna(subset=[group_col, metric_col])
        if clean_data.empty:
            return None, None
        
        comparison = clean_data.groupby(group_col)[metric_col].mean().reset_index()
        comparison = comparison.sort_values(metric_col, ascending=False).head(10)
        
        chart_data = {
            "type": "bar",
            "title": f"{metric_col.replace('_', ' ').title()} by {title_suffix}",
            "xAxis": {
                "label": title_suffix,
                "type": "category",
                "categories": list(comparison[group_col])
            },
            "yAxis": {
                "label": metric_col.replace('_', ' ').title(),
                "type": "value"
            },
            "series": [{
                "name": metric_col.replace('_', ' ').title(),
                "type": "bar",
                "data": [float(val) for val in comparison[metric_col]],
                "color": "#007bff"
            }]
        }
        
        return chart_data, "bar"

    def create_temporal_trend_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create temporal trend chart based on query"""
        if 'Year' not in data.columns:
            return None, None
        
        # Determine metric based on query
        metric_col = None
        if 'rainfall' in query:
            metric_col = 'Annual_Rainfall_mm'
        elif 'recharge' in query:
            metric_col = 'Annual_Recharge_MCM'
        elif 'extraction' in query:
            metric_col = 'Annual_Extraction_MCM'
        elif 'groundwater' in query:
            metric_col = 'Groundwater_Level_m'
        elif 'population' in query:
            metric_col = 'Population'
        
        if not metric_col or metric_col not in data.columns:
            return None, None
        
        clean_data = data.dropna(subset=['Year', metric_col])
        if clean_data.empty:
            return None, None
        
        # Group by year and calculate mean
        trend_data = clean_data.groupby('Year')[metric_col].mean().reset_index()
        
        chart_data = {
            "type": "line",
            "title": f"{metric_col.replace('_', ' ').title()} Trend Over Time",
            "xAxis": {
                "label": "Year",
                "type": "category"
            },
            "yAxis": {
                "label": metric_col.replace('_', ' ').title(),
                "type": "value"
            },
            "series": [{
                "name": metric_col.replace('_', ' ').title(),
                "type": "line",
                "data": [
                    {"x": int(row['Year']), "y": float(row[metric_col])} 
                    for _, row in trend_data.iterrows()
                ]
            }]
        }
        
        return chart_data, "line"

    def create_multi_state_rainfall_comparison_json(self, data: pd.DataFrame, states_list: List[str]) -> tuple[Optional[Dict], Optional[str]]:
        """Create BAR chart comparing rainfall between ONLY the specified states"""
        print(f"[DEBUG] Creating rainfall comparison for ONLY these states: {states_list}")
        if 'Annual_Rainfall_mm' not in data.columns or 'State' not in data.columns:
            return None, None
        clean_data = data.dropna(subset=['Annual_Rainfall_mm', 'State'])
        clean_data = clean_data[clean_data['State'].isin(states_list)]
        if clean_data.empty:
            print(f"[DEBUG] No data found for states: {states_list}")
            return None, None
        state_rainfall = clean_data.groupby('State')['Annual_Rainfall_mm'].mean().reset_index()
        state_rainfall = state_rainfall[state_rainfall['State'].isin(states_list)]
        chart_data = {
            "type": "bar",
            "title": f"Rainfall Comparison: {' vs '.join(states_list)}",
            "xAxis": {
                "label": "State",
                "type": "category",
                "categories": state_rainfall['State'].tolist()
            },
            "yAxis": {
                "label": "Annual Rainfall (mm)",
                "type": "value"
            },
            "series": [{
                "name": "Average Rainfall",
                "type": "bar",
                "data": [round(float(val), 1) for val in state_rainfall['Annual_Rainfall_mm']],
                "color": "#007bff"
            }]
        }
        print(f"[DEBUG] Created comparison for {len(state_rainfall)} states: {state_rainfall['State'].tolist()}")
        return chart_data, "bar"

    # Keep existing methods from original code
    def extract_states_from_query(self, query: str) -> List[str]:
        """Extract multiple state names from query with EXACT matching"""
        query_lower = query.lower().strip()
        states = self.df['State'].dropna().unique()
        mentioned_states: List[str] = []
        # Precise variations and abbreviations
        state_variations = {
            'gujarat': 'Gujarat',
            'punjab': 'Punjab', 
            'assam': 'Assam',
            'maharashtra': 'Maharashtra',
            'karnataka': 'Karnataka',
            'tamil nadu': 'Tamil Nadu',
            'tamilnadu': 'Tamil Nadu',
            'west bengal': 'West Bengal',
            'westbengal': 'West Bengal',
            'uttar pradesh': 'Uttar Pradesh',
            'up': 'Uttar Pradesh',
            'madhya pradesh': 'Madhya Pradesh',
            'mp': 'Madhya Pradesh',
            'andhra pradesh': 'Andhra Pradesh',
            'ap': 'Andhra Pradesh',
            'rajasthan': 'Rajasthan',
            'bihar': 'Bihar',
            'odisha': 'Odisha',
            'orissa': 'Odisha',
            'kerala': 'Kerala',
            'haryana': 'Haryana',
            'delhi': 'Delhi',
            'jammu and kashmir': 'Jammu & Kashmir',
            'j&k': 'Jammu & Kashmir',
            'himachal pradesh': 'Himachal Pradesh',
            'hp': 'Himachal Pradesh',
            'uttarakhand': 'Uttarakhand',
            'chhattisgarh': 'Chhattisgarh',
            'jharkhand': 'Jharkhand',
            'telangana': 'Telangana'
        }
        import re
        # Exact matches for state names from dataset
        for state in states:
            if pd.notna(state) and isinstance(state, str):
                pattern = r'\b' + re.escape(state.lower()) + r'\b'
                if re.search(pattern, query_lower):
                    if state not in mentioned_states:
                        mentioned_states.append(state)
        # Variations/abbreviations with boundaries
        for variation, full_name in state_variations.items():
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, query_lower) and full_name in states:
                if full_name not in mentioned_states:
                    mentioned_states.append(full_name)
        # DEBUG
        available_states = [str(s) for s in states if pd.notna(s)]
        print(f"[DEBUG] Query: '{query}' -> Extracted states: {mentioned_states}")
        print(f"[DEBUG] Available states in dataset: {available_states[:10]}...")
        return mentioned_states
    
    def extract_year_from_query(self, query: str) -> Optional[int]:
        """Extract year from query"""
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        year = int(year_match.group()) if year_match else None
        
        if not year:
            year_phrase_match = re.search(r'\bin\s+(19|20)\d{2}\b', query.lower())
            if year_phrase_match:
                year = int(year_phrase_match.group(1) + year_phrase_match.group(2))
        
        print(f"[DEBUG] Extracted year from query '{query}': {year}")
        return year

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
            
            # Category comparison - BAR CHART  
            elif 'compare' in query_lower and 'category' in query_lower:
                return self.create_category_comparison_bar_chart_json(data, query_lower)
            
            # Trend analysis - LINE CHART
            elif 'trend' in query_lower and 'Year' in data.columns:
                return self.create_trend_line_chart_json(data)
            
            # General comparison queries - BAR CHART
            elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'between']):
                return self.create_comparison_bar_chart_json(data, query_lower)
            
            # State-specific average queries - BAR CHART or LINE CHART
            elif 'average' in query_lower and 'State' in data.columns:
                return self.create_average_chart_json(data, query_lower)
            
            return None, None
                        
        except Exception as e:
            print(f"[DEBUG] Chart creation error: {str(e)}")
            return None, None

    def create_distribution_pie_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create PIE chart for distribution/breakdown queries"""
        print(f"[DEBUG] Creating distribution PIE chart")
        
        if 'category' in query and 'Category' in data.columns:
            clean_data = data.dropna(subset=['Category'])
            if clean_data.empty:
                return None, None
                
            distribution = clean_data['Category'].value_counts()
            
            category_colors = {
                'Safe': '#28a745',
                'Semi-Critical': '#ffc107', 
                'Critical': '#fd7e14',
                'Over-Exploited': '#dc3545',
                'Over-Critical': '#dc3545'
            }
            
            chart_data = {
                "type": "pie",
                "title": "Distribution of Groundwater Categories",
                "series": [{
                    "name": "Category Distribution",
                    "type": "pie",
                    "data": [
                        {
                            "name": category,
                            "value": int(count),
                            "color": category_colors.get(category, '#007bff')
                        }
                        for category, count in distribution.items()
                    ]
                }]
            }
            
        elif 'state' in query and 'State' in data.columns:
            distribution = data['State'].value_counts().head(10)
            
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

    def create_category_comparison_bar_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create BAR chart for category comparison"""
        print(f"[DEBUG] Creating category comparison BAR chart")
        
        if 'Category' not in data.columns:
            print("[DEBUG] No Category column found")
            return None, None
        
        clean_data = data.dropna(subset=['Category']).copy()
        
        if clean_data.empty:
            print("[DEBUG] No valid category data after cleaning")
            return None, None
        
        states = clean_data['State'].dropna().unique()
        all_categories = clean_data['Category'].dropna().unique()
        
        category_colors = {
            'Safe': '#28a745',           
            'Semi-Critical': '#ffc107',  
            'Critical': '#fd7e14',       
            'Over-Exploited': '#dc3545', 
            'Over-Critical': '#dc3545'   
        }

        chart_data = {
            "type": "bar",
            "title": f"Category Comparison - {', '.join(states)}" if len(states) <= 3 else "Category by State",
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

        for category in sorted(all_categories):
            series_data = []
            
            for state in states:
                count = len(clean_data[
                    (clean_data['State'] == state) & 
                    (clean_data['Category'] == category)
                ])
                series_data.append(count)
            
            chart_data["series"].append({
                "name": category,
                "type": "bar",
                "data": series_data,
                "color": category_colors.get(category, '#007bff')
            })
        
        print(f"[DEBUG] Category comparison BAR chart created with {len(chart_data['series'])} series")
        return chart_data, "bar"

    def create_trend_line_chart_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create LINE chart for trend analysis"""
        print(f"[DEBUG] Creating trend LINE chart")
        
        if 'Year' not in data.columns:
            print("[DEBUG] No Year column found")
            return None, None
            
        numeric_cols = ['Groundwater_Level_m', 'Annual_Recharge_MCM', 'Annual_Extraction_MCM', 'Annual_Rainfall_mm']
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
        
        if 'State' in data.columns and data['State'].nunique() > 1:
            chart_data["title"] += " by State"
            
            for state in data['State'].dropna().unique():
                state_data = data[data['State'] == state]
                if state_data.empty:
                    continue
                    
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

    def create_comparison_bar_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create BAR chart for general comparisons"""
        print(f"[DEBUG] Creating comparison BAR chart")
        
        metric_col = None
        chart_title = ""
        
        if 'groundwater' in query.lower() and 'level' in query.lower():
            metric_col = 'Groundwater_Level_m'
            chart_title = "Groundwater Level Comparison"
        elif 'recharge' in query.lower():
            metric_col = 'Annual_Recharge_MCM'
            chart_title = "Recharge Comparison"
        elif 'extraction' in query.lower():
            metric_col = 'Annual_Extraction_MCM'
            chart_title = "Extraction Comparison"
        elif 'rainfall' in query.lower():
            metric_col = 'Annual_Rainfall_mm'
            chart_title = "Rainfall Comparison"
        elif 'population' in query.lower():
            metric_col = 'Population'
            chart_title = "Population Comparison"
        elif 'category' in query.lower():
            return self.create_category_comparison_bar_chart_json(data, query)
        
        if not metric_col:
            numeric_cols = ['Groundwater_Level_m', 'Annual_Recharge_MCM', 'Annual_Extraction_MCM', 'Annual_Rainfall_mm']
            available_cols = [col for col in numeric_cols if col in data.columns and data[col].notna().sum() > 0]
            
            if available_cols:
                metric_col = available_cols[0]
                chart_title = f'{metric_col.replace("_", " ").title()} Comparison'
        
        if not metric_col or metric_col not in data.columns or 'State' not in data.columns:
            print("[DEBUG] Insufficient data for comparison chart")
            return None, None
        
        clean_data = data.dropna(subset=[metric_col, 'State'])
        
        if clean_data.empty:
            print("[DEBUG] No valid data after cleaning")
            return None, None
        
        comparison_data = clean_data.groupby('State')[metric_col].mean().reset_index()
        comparison_data = comparison_data.sort_values(metric_col, ascending=False)
        
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
        return chart_data, "bar"

    def create_average_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create appropriate chart for average queries"""
        print(f"[DEBUG] Creating average chart")
        
        numeric_cols = ['Groundwater_Level_m', 'Annual_Recharge_MCM', 'Annual_Extraction_MCM', 'Annual_Rainfall_mm', 'Population']
        
        metric_col = 'Groundwater_Level_m'  # default
        if 'recharge' in query:
            metric_col = 'Annual_Recharge_MCM' if 'Annual_Recharge_MCM' in data.columns else metric_col
        elif 'extraction' in query:
            metric_col = 'Annual_Extraction_MCM' if 'Annual_Extraction_MCM' in data.columns else metric_col
        elif 'rainfall' in query:
            metric_col = 'Annual_Rainfall_mm' if 'Annual_Rainfall_mm' in data.columns else metric_col
        elif 'population' in query:
            metric_col = 'Population' if 'Population' in data.columns else metric_col
        
        if metric_col not in data.columns or data[metric_col].notna().sum() == 0:
            print(f"[DEBUG] No data available for metric: {metric_col}")
            return None, None
        
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

    def clean_response(self, response: str) -> str:
        """Clean up agent response"""
        import re
        
        response = re.sub(r'Action:\s*python_repl_ast.*?(?=Thought:|Final Answer:|$)', '', response, flags=re.DOTALL)
        response = re.sub(r'Thought:.*?(?=Final Answer:|Thought:|$)', '', response, flags=re.DOTALL)
        
        final_answer_match = re.search(r'Final Answer:\s*(.*?)(?=Action:|Thought:|$)', response, re.DOTALL)
        if final_answer_match:
            response = final_answer_match.group(1).strip()
        
        response = re.sub(r'There are \d+ rows?.*?(?=\.|\n|$)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'The dataframe.*?(?=\.|\n|$)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Based on the data.*?(?=\.|\n|$)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'In the dataset.*?(?=\.|\n|$)', '', response, flags=re.IGNORECASE)
        
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()
        
        return response
    
    def is_greeting_query(self, query: str) -> bool:
        """Check if query is a simple greeting"""
        query_lower = query.lower().strip()
        simple_greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
                           'how are you', 'how you doing', 'what\'s up', 'greetings']
        return query_lower in simple_greetings
    
    def is_groundwater_query(self, query: str) -> bool:
        """Check if query is related to groundwater data"""
        query_lower = query.lower()
        
        # Handle common typos and variations
        query_lower = query_lower.replace('grndwater', 'groundwater')
        query_lower = query_lower.replace('grounwater', 'groundwater')
        query_lower = query_lower.replace('ground water', 'groundwater')
        query_lower = query_lower.replace('ground-water', 'groundwater')
        
        groundwater_keywords = [
            'groundwater', 'water', 'aquifer', 'recharge', 'extraction', 'rainfall',
            'level', 'depth', 'well', 'borewell', 'tube well', 'water table',
            'safe', 'critical', 'semi-critical', 'over-exploited', 'saline',
            'state', 'district', 'city', 'population', 'area', 'year'
        ]
        
        # Check if query contains groundwater-related keywords
        return any(keyword in query_lower for keyword in groundwater_keywords) 
    
    def requires_visualization(self, query: str) -> bool:
        """Check if query requires visualization - FIXED VERSION"""
        query_lower = query.lower().strip()
        # ML prediction visualization patterns
        ml_viz_patterns = [
            'predict', 'forecast', 'future', 'trends', 'chart future',
            'visualize', 'show forecast', 'plot', '2030', '2035', '2040'
        ]
        for pattern in ml_viz_patterns:
            if pattern in query_lower:
                print(f"[DEBUG] Matched ML visualization pattern: '{pattern}'")
                return True
        # Specific visualization patterns
        specific_patterns = [
            'show category distribution',
            'category distribution', 
            'show population by state',
            'population by state',
            'show distribution',
            'show breakdown'
        ]
        for pattern in specific_patterns:
            if pattern in query_lower:
                print(f"[DEBUG] Matched specific visualization pattern: '{pattern}'")
                return True
        # Direct chart requests
        chart_keywords = ['chart', 'graph', 'plot', 'visualization', 'visualize', 'visualise']
        if any(keyword in query_lower for keyword in chart_keywords):
            print(f"[DEBUG] Matched direct chart keyword")
            return True
        # Only require visualization for explicit comparison requests on metrics
        comparison_keywords = ['compare', 'comparison', 'vs', 'versus']
        metric_keywords = ['level', 'rainfall', 'recharge', 'extraction', 'population']
        if any(k in query_lower for k in comparison_keywords) and any(m in query_lower for m in metric_keywords):
            print(f"[DEBUG] Matched comparison with data visualization")
            return True
        # Trend analysis
        if 'trend' in query_lower:
            print(f"[DEBUG] Matched trend visualization")
            return True
        print(f"[DEBUG] No visualization pattern matched for: '{query}'")
        return False
    
    def handle_greeting_query(self, query: str) -> str:
        """Handle greeting queries"""
        return "Hello! I'm your enhanced groundwater data assistant. You can ask me about groundwater levels, recharge rates, extraction data, population, rainfall, and trends across different states, districts, and cities. I can also help with location-based queries using your current position!"
    
    def generate_viz_response(self, query: str, data: pd.DataFrame, location_info: Optional[Dict]) -> str:
        """Generate appropriate text response for visualization queries"""
        query_lower = query.lower()
        
        location_prefix = ""
        if location_info and location_info.get('closest_location'):
            closest = location_info['closest_location']
            location_prefix = f"Near your location ({closest['city']}, {closest['district']}, {closest['state']}): "
        
        if data.empty:
            return f"{location_prefix}No data found for visualization."
        
        # Category distribution summary
        if 'category' in query_lower and 'distribution' in query_lower:
            if 'Category' in data.columns:
                categories = data['Category'].value_counts()
                total = len(data)
                response = f"{location_prefix}Groundwater Category Distribution ({total} locations):\n"
                for cat, count in categories.items():
                    pct = (count / total) * 100 if total else 0
                    response += f"• {cat}: {count} ({pct:.1f}%)\n"
                return response.strip()
        
        # Population by state summary
        if 'population' in query_lower and 'state' in query_lower:
            if 'Population' in data.columns and 'State' in data.columns:
                state_pop = data.groupby('State')['Population'].sum().sort_values(ascending=False)
                response = f"{location_prefix}Population by State (Top 10):\n"
                for state, pop in state_pop.head(10).items():
                    response += f"• {state}: {int(pop):,}\n"
                return response.strip()
        
        # Generic
        return f"{location_prefix}Here's the visualization for your query. The chart shows {len(data)} data points."

    def enhance_input(self, query: str) -> str:
        """Enhance input by fixing typos, expanding abbreviations, and improving query clarity"""
        if not query or not query.strip():
            return query
            
        # Convert to lowercase for processing
        enhanced = query.lower().strip()
        
        # Fix common typos and variations
        typo_fixes = {
            'grndwater': 'groundwater',
            'grounwater': 'groundwater', 
            'ground water': 'groundwater',
            'ground-water': 'groundwater',
            'underground water': 'groundwater',
            'subsurface water': 'groundwater',
            'aquafer': 'aquifer',
            'aquaifer': 'aquifer',
            'rechage': 'recharge',
            'rechare': 'recharge',
            'extracton': 'extraction',
            'extractin': 'extraction',
            'rainfal': 'rainfall',
            'rainfal': 'rainfall',
            'populaton': 'population',
            'populatin': 'population',
            'distric': 'district',
            'distict': 'district',
            'punjab': 'punjab',
            'punjab': 'punjab',
            'maharashtra': 'maharashtra',
            'maharashtr': 'maharashtra',
            'karnataka': 'karnataka',
            'karnatak': 'karnataka',
            'tamil nadu': 'tamil nadu',
            'tamilnadu': 'tamil nadu',
            'west bengal': 'west bengal',
            'westbengal': 'west bengal',
            'uttar pradesh': 'uttar pradesh',
            'up': 'uttar pradesh',
            'bihar': 'bihar',
            'rajasthan': 'rajasthan',
            'gujarat': 'gujarat',
            'madhya pradesh': 'madhya pradesh',
            'mp': 'madhya pradesh',
            'andhra pradesh': 'andhra pradesh',
            'ap': 'andhra pradesh',
            'odisha': 'odisha',
            'orissa': 'odisha',
            'kerala': 'kerala',
            'assam': 'assam',
            'haryana': 'haryana',
            'delhi': 'delhi',
            'nct': 'delhi',
            'jammu and kashmir': 'jammu and kashmir',
            'j&k': 'jammu and kashmir',
            'himachal pradesh': 'himachal pradesh',
            'hp': 'himachal pradesh',
            'uttarakhand': 'uttarakhand',
            'uk': 'uttarakhand',
            'chhattisgarh': 'chhattisgarh',
            'jharkhand': 'jharkhand',
            'manipur': 'manipur',
            'meghalaya': 'meghalaya',
            'mizoram': 'mizoram',
            'nagaland': 'nagaland',
            'sikkim': 'sikkim',
            'tripura': 'tripura',
            'arunachal pradesh': 'arunachal pradesh',
            'goa': 'goa',
            'telangana': 'telangana',
            'ts': 'telangana'
        }
        
        # Apply typo fixes
        for typo, correct in typo_fixes.items():
            enhanced = enhanced.replace(typo, correct)
        
        # Expand common abbreviations
        abbreviations = {
            'gw': 'groundwater',
            'gwl': 'groundwater level',
            'wl': 'water level',
            'wt': 'water table',
            'aq': 'aquifer',
            'rech': 'recharge',
            'ext': 'extraction',
            'rf': 'rainfall',
            'pop': 'population',
            'dist': 'district',
            'st': 'state',
            'cty': 'city',
            'yr': 'year',
            'avg': 'average',
            'tot': 'total',
            'max': 'maximum',
            'min': 'minimum',
            'pct': 'percent',
            '%': 'percent',
            'mcm': 'million cubic meters',
            'mm': 'millimeters',
            'm': 'meters',
            'km': 'kilometers',
            'sq km': 'square kilometers',
            'km2': 'square kilometers',
            'safe': 'safe category',
            'critical': 'critical category',
            'semi-critical': 'semi-critical category',
            'over-exploited': 'over-exploited category',
            'saline': 'saline water',
            'fresh': 'fresh water',
            'non-saline': 'non-saline water'
        }
        
        # Apply abbreviation expansions
        for abbr, expansion in abbreviations.items():
            # Use word boundaries to avoid partial matches
            import re
            pattern = r'\b' + re.escape(abbr) + r'\b'
            enhanced = re.sub(pattern, expansion, enhanced, flags=re.IGNORECASE)
        
        # Fix common grammatical issues
        grammar_fixes = {
            'what is the': 'what is',
            'what are the': 'what are',
            'show me the': 'show',
            'give me the': 'give',
            'tell me the': 'tell',
            'can you tell me': 'tell',
            'can you show me': 'show',
            'can you give me': 'give',
            'i want to know': 'tell',
            'i need to know': 'tell',
            'i would like to know': 'tell'
        }
        
        for phrase, replacement in grammar_fixes.items():
            if enhanced.startswith(phrase):
                enhanced = enhanced.replace(phrase, replacement, 1)
        
        # Add missing question words
        if not any(enhanced.startswith(q) for q in ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'show', 'tell', 'give', 'list', 'find', 'get']):
            if 'groundwater' in enhanced or 'water' in enhanced:
                enhanced = 'what is ' + enhanced
            elif 'level' in enhanced or 'data' in enhanced:
                enhanced = 'show ' + enhanced
            else:
                enhanced = 'tell me about ' + enhanced
        
        # Clean up extra spaces
        enhanced = ' '.join(enhanced.split())
        
        return enhanced

    def handle_general_query(self, query: str) -> str:
        """Handle general queries while highlighting groundwater expertise"""
        query_lower = query.lower()
        
        # Biology questions
        if any(word in query_lower for word in ['cell', 'mitochondria', 'powerhouse', 'biology', 'dna', 'protein']):
            if 'powerhouse' in query_lower and 'cell' in query_lower:
                return "The powerhouse of the cell is the mitochondria! It's called the powerhouse because it produces ATP (adenosine triphosphate), which is the energy currency of the cell. The mitochondria convert nutrients into energy through cellular respiration.\n\nBy the way, I have special expertise in groundwater data analysis. If you're interested in water-related topics, I can help you explore groundwater levels, recharge rates, extraction data, and water quality across different regions!"
        
        # General science questions
        elif any(word in query_lower for word in ['science', 'physics', 'chemistry', 'math', 'mathematics']):
            return f"I'd be happy to help with that {query_lower.split()[0]} question! However, I have specialized expertise in groundwater data analysis. I can provide detailed insights about water resources, aquifer conditions, recharge patterns, and environmental data across different states and regions. Feel free to ask me about groundwater-related topics for the most comprehensive analysis!"
        
        # Weather/climate questions
        elif any(word in query_lower for word in ['weather', 'climate', 'temperature', 'rain', 'precipitation']):
            return f"I can help with weather and climate questions! Interestingly, I have extensive data on rainfall patterns and their impact on groundwater recharge across different regions. I can analyze annual rainfall trends, seasonal variations, and their correlation with groundwater levels. Would you like to explore rainfall data for specific states or regions?"
        
        # Geography questions
        elif any(word in query_lower for word in ['country', 'state', 'city', 'location', 'geography', 'map']):
            return f"I can help with geography questions! I have detailed data about states, districts, and cities across India, including their population, area, and environmental characteristics. I can provide insights about groundwater conditions, water availability, and environmental factors for different locations. Would you like to explore groundwater data for specific regions?"
        
        # General knowledge questions
        else:
            return f"I can help answer that question! However, I have specialized expertise in groundwater data analysis and water resource management. I can provide detailed insights about:\n\n• Groundwater levels and trends\n• Water recharge and extraction patterns\n• Environmental conditions across different regions\n• Population and demographic data\n• Rainfall and climate patterns\n\nFeel free to ask me about groundwater-related topics for the most comprehensive analysis, or I can help with your general question too!"


# Initialize assistant globally
assistant = None

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on startup"""
    global assistant
    csv_path = "data/indian_groundwater.csv"
    assistant = GroundwaterAssistant(csv_path)
    print("Enhanced Groundwater Assistant initialized")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "INGRES Enhanced Groundwater Data Assistant API", "version": "2.0.0"}

# @app.get("/summary", response_model=DataSummary)
# async def get_summary():
#     """Get enhanced data summary"""
#     if not assistant or assistant.df.empty:
#         raise HTTPException(status_code=500, detail="Data not available")
    
#     df = assistant.df
    
#     return DataSummary(
#         total_records=len(df),
#         states_count=df['State'].nunique() if 'State' in df.columns else 0,
#         districts_count=df['District'].nunique() if 'District' in df.columns else 0,
#         cities_count=df['City'].nunique() if 'City' in df.columns else 0,
#         years_range=(int(df['Year'].min()), int(df['Year'].max())) if 'Year' in df.columns else (0, 0),
#         available_years=sorted(df['Year'].dropna().unique().astype(int).tolist()) if 'Year' in df.columns else [],
#         available_states=sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else [],
#         available_districts=sorted(df['District'].dropna().unique().tolist()) if 'District' in df.columns else [],
#         available_cities=sorted(df['City'].dropna().unique().tolist()) if 'City' in df.columns else []
#     )

@app.post("/query", response_model=Union[QueryResponse, SimpleResponse])
async def query_data(request: QueryRequest):
    """Process enhanced user query with smart response formatting"""
    if not assistant:
        raise HTTPException(status_code=500, detail="Assistant not initialized")
    
    if assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Check if visualization is required
        requires_viz = assistant.requires_visualization(request.question)
        
        result = assistant.query_data(request.question, request.latitude, request.longitude)
        
        if requires_viz:
            # Return full QueryResponse for visualization queries
            return QueryResponse(**result)
        else:
            # Return simple response for non-visualization queries
            return SimpleResponse(response=result["response"])
            
    except Exception as e:
        print(f"[ERROR] Query processing failed: {str(e)}")
        if assistant and hasattr(assistant, 'requires_visualization') and assistant.requires_visualization(request.question):
            return QueryResponse(
                response="I'm having trouble processing that visualization request. Please try rephrasing it or ask something more specific about the groundwater data.",
                data=None,
                chart=None,
                chart_type=None,
                location_info=None
            )
        else:
            return SimpleResponse(
                response="I'm having trouble processing that question. Please try rephrasing it or ask something more specific about the groundwater data."
            )

@app.post("/location-data")
async def get_location_data(request: LocationRequest):
    """Get data for specific location"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    try:
        nearby_data = assistant.find_nearest_locations(
            request.latitude, 
            request.longitude, 
            request.radius_km
        )
        
        if nearby_data.empty:
            return {"message": "No data found within specified radius", "data": []}
        
        return {
            "message": f"Found {len(nearby_data)} locations within {request.radius_km}km",
            "data": nearby_data.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting location data: {str(e)}")


@app.get("/debug-data")
async def debug_data():
    """Debug endpoint to check data loading"""
    if not assistant or assistant.df.empty:
        return {"error": "No data loaded"}
    
    df = assistant.df
    
    return {
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "category_values": df['Category'].value_counts().to_dict() if 'Category' in df.columns else "No Category column",
        "sample_data": df.head(3).to_dict('records'),
        "population_stats": {
            "total": int(df['Population'].sum()) if 'Population' in df.columns and pd.notna(df['Population']).any() else "No Population column",
            "by_state": df.groupby('State')['Population'].sum().head(5).to_dict() if 'State' in df.columns and 'Population' in df.columns else "Cannot group"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
