"""
AI-Powered Retail Intelligence Platform
========================================
A comprehensive solution for retail decision-making with:
- Market trend analysis & forecasting
- Dynamic pricing intelligence
- Customer demand prediction
- Competitor analysis
- Business insights generation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MarketIntelligenceEngine:
    """AI-powered market analysis and forecasting engine"""
    
    def __init__(self):
        self.historical_data = None
        self.forecasts = {}
        
    def analyze_market_trends(self, sales_data: pd.DataFrame) -> Dict:
        """Analyze market trends using time series analysis"""
        trends = {
            'overall_trend': self._calculate_trend(sales_data),
            'seasonality': self._detect_seasonality(sales_data),
            'growth_rate': self._calculate_growth_rate(sales_data),
            'volatility': self._calculate_volatility(sales_data)
        }
        return trends
    
    def _calculate_trend(self, data: pd.DataFrame) -> str:
        """Calculate overall market trend"""
        if len(data) < 2:
            return "Insufficient data"
        
        # Simple linear regression
        x = np.arange(len(data))
        y = data['sales'].values
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.05:
            return "Strong Upward"
        elif slope > 0:
            return "Moderate Upward"
        elif slope > -0.05:
            return "Stable"
        else:
            return "Declining"
    
    def _detect_seasonality(self, data: pd.DataFrame) -> Dict:
        """Detect seasonal patterns"""
        if 'date' in data.columns:
            data['month'] = pd.to_datetime(data['date']).dt.month
            monthly_avg = data.groupby('month')['sales'].mean()
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()
            
            return {
                'peak_month': int(peak_month),
                'low_month': int(low_month),
                'seasonality_strength': float(monthly_avg.std() / monthly_avg.mean())
            }
        return {'seasonality_detected': False}
    
    def _calculate_growth_rate(self, data: pd.DataFrame) -> float:
        """Calculate compound growth rate"""
        if len(data) < 2:
            return 0.0
        
        first_value = data['sales'].iloc[0]
        last_value = data['sales'].iloc[-1]
        periods = len(data)
        
        if first_value <= 0:
            return 0.0
            
        growth_rate = ((last_value / first_value) ** (1/periods) - 1) * 100
        return round(growth_rate, 2)
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate market volatility"""
        returns = data['sales'].pct_change().dropna()
        volatility = returns.std() * 100
        return round(volatility, 2)
    
    def forecast_demand(self, historical_data: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
        """Forecast future demand using exponential smoothing"""
        sales = historical_data['sales'].values
        
        # Triple exponential smoothing (Holt-Winters)
        alpha, beta, gamma = 0.3, 0.1, 0.1
        
        # Initialize
        level = sales[0]
        trend = 0
        seasonal = np.zeros(12)
        
        forecast = []
        
        for t in range(len(sales)):
            prev_level = level
            level = alpha * sales[t] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        # Generate forecasts
        for i in range(periods):
            forecast_value = level + (i + 1) * trend
            forecast.append(max(0, forecast_value))  # Ensure non-negative
        
        forecast_df = pd.DataFrame({
            'period': range(1, periods + 1),
            'forecasted_sales': forecast,
            'confidence_lower': [f * 0.85 for f in forecast],
            'confidence_upper': [f * 1.15 for f in forecast]
        })
        
        return forecast_df


class PricingIntelligence:
    """Dynamic pricing optimization engine"""
    
    def __init__(self):
        self.pricing_history = {}
        
    def analyze_price_elasticity(self, price_data: pd.DataFrame) -> Dict:
        """Calculate price elasticity of demand"""
        if len(price_data) < 2:
            return {'elasticity': 0, 'recommendation': 'Need more data'}
        
        # Calculate percentage changes
        price_change = price_data['price'].pct_change()
        demand_change = price_data['quantity'].pct_change()
        
        # Remove NaN and infinity values
        valid_data = pd.DataFrame({
            'price_change': price_change,
            'demand_change': demand_change
        }).replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(valid_data) < 2:
            elasticity = -1.0
        else:
            # Price elasticity = % change in demand / % change in price
            elasticity = (valid_data['demand_change'] / valid_data['price_change']).mean()
        
        # Classify elasticity
        if abs(elasticity) > 1:
            category = "Elastic"
            recommendation = "Price sensitive - small price changes have large impact"
        elif abs(elasticity) < 1:
            category = "Inelastic"
            recommendation = "Price insensitive - can increase prices moderately"
        else:
            category = "Unit Elastic"
            recommendation = "Proportional response - balance price and volume"
        
        return {
            'elasticity': round(elasticity, 2),
            'category': category,
            'recommendation': recommendation
        }
    
    def optimize_pricing(self, current_price: float, cost: float, 
                        demand_forecast: float, competitor_prices: List[float]) -> Dict:
        """Recommend optimal pricing strategy"""
        
        # Calculate key metrics
        avg_competitor_price = np.mean(competitor_prices) if competitor_prices else current_price
        min_competitor_price = min(competitor_prices) if competitor_prices else current_price
        max_competitor_price = max(competitor_prices) if competitor_prices else current_price
        
        # Pricing strategies
        strategies = {
            'cost_plus': cost * 1.3,  # 30% margin
            'competitive': avg_competitor_price * 0.98,  # 2% below average
            'premium': max_competitor_price * 1.05,  # 5% above max
            'penetration': min_competitor_price * 0.92,  # 8% below min
            'value_based': cost * 1.5  # 50% margin for high value
        }
        
        # Calculate expected revenue for each strategy
        revenue_estimates = {}
        for strategy, price in strategies.items():
            # Simple demand estimation (higher price = lower demand)
            price_factor = current_price / price if price > 0 else 1
            estimated_demand = demand_forecast * price_factor
            revenue = price * estimated_demand
            profit_margin = ((price - cost) / price) * 100 if price > 0 else 0
            
            revenue_estimates[strategy] = {
                'price': round(price, 2),
                'estimated_demand': round(estimated_demand, 2),
                'estimated_revenue': round(revenue, 2),
                'profit_margin': round(profit_margin, 2)
            }
        
        # Find best strategy (highest revenue with acceptable margin)
        best_strategy = max(
            revenue_estimates.items(),
            key=lambda x: x[1]['estimated_revenue'] if x[1]['profit_margin'] > 15 else 0
        )
        
        return {
            'current_price': current_price,
            'recommended_strategy': best_strategy[0],
            'recommended_price': best_strategy[1]['price'],
            'all_strategies': revenue_estimates,
            'market_position': self._classify_market_position(current_price, avg_competitor_price)
        }
    
    def _classify_market_position(self, your_price: float, avg_competitor: float) -> str:
        """Classify pricing position in market"""
        ratio = your_price / avg_competitor if avg_competitor > 0 else 1
        
        if ratio > 1.1:
            return "Premium positioned"
        elif ratio > 0.95:
            return "Competitively positioned"
        else:
            return "Value positioned"


class CustomerInsightsEngine:
    """AI-powered customer behavior analysis"""
    
    def segment_customers(self, customer_data: pd.DataFrame) -> Dict:
        """Segment customers using RFM analysis"""
        
        # Calculate RFM scores
        rfm_data = customer_data.copy()
        
        # Recency score (lower is better)
        rfm_data['recency_score'] = pd.qcut(
            rfm_data['days_since_purchase'], 
            q=4, 
            labels=[4, 3, 2, 1],
            duplicates='drop'
        )
        
        # Frequency score (higher is better)
        rfm_data['frequency_score'] = pd.qcut(
            rfm_data['purchase_count'].rank(method='first'), 
            q=4, 
            labels=[1, 2, 3, 4],
            duplicates='drop'
        )
        
        # Monetary score (higher is better)
        rfm_data['monetary_score'] = pd.qcut(
            rfm_data['total_spent'].rank(method='first'), 
            q=4, 
            labels=[1, 2, 3, 4],
            duplicates='drop'
        )
        
        # Convert to numeric
        rfm_data['recency_score'] = pd.to_numeric(rfm_data['recency_score'])
        rfm_data['frequency_score'] = pd.to_numeric(rfm_data['frequency_score'])
        rfm_data['monetary_score'] = pd.to_numeric(rfm_data['monetary_score'])
        
        # Create segments
        rfm_data['segment'] = rfm_data.apply(self._assign_segment, axis=1)
        
        # Analyze segments
        segments = rfm_data['segment'].value_counts().to_dict()
        segment_value = rfm_data.groupby('segment')['total_spent'].sum().to_dict()
        
        return {
            'segments': segments,
            'segment_value': segment_value,
            'recommendations': self._generate_segment_recommendations(segments)
        }
    
    def _assign_segment(self, row) -> str:
        """Assign customer to segment based on RFM scores"""
        r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
        
        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        elif r >= 3 and f >= 2:
            return "Loyal Customers"
        elif r >= 3 and m >= 3:
            return "Big Spenders"
        elif r >= 3:
            return "Promising"
        elif f >= 3:
            return "Need Attention"
        elif m >= 3:
            return "At Risk"
        else:
            return "Lost"
    
    def _generate_segment_recommendations(self, segments: Dict) -> Dict:
        """Generate marketing recommendations for each segment"""
        recommendations = {
            "Champions": "Reward with exclusive offers, early access to new products",
            "Loyal Customers": "Upsell premium products, loyalty program benefits",
            "Big Spenders": "Cross-sell complementary items, VIP treatment",
            "Promising": "Nurture with targeted campaigns, onboarding programs",
            "Need Attention": "Re-engagement campaigns, personalized recommendations",
            "At Risk": "Win-back offers, feedback surveys, special discounts",
            "Lost": "Aggressive win-back campaigns or accept loss"
        }
        
        active_recommendations = {
            seg: recommendations.get(seg, "Analyze further")
            for seg in segments.keys()
        }
        
        return active_recommendations


class CompetitorAnalyzer:
    """Competitive intelligence and analysis"""
    
    def analyze_competitive_position(self, your_metrics: Dict, 
                                    competitor_metrics: List[Dict]) -> Dict:
        """Analyze competitive positioning"""
        
        # Calculate market share
        total_market = your_metrics['revenue'] + sum(c['revenue'] for c in competitor_metrics)
        your_share = (your_metrics['revenue'] / total_market * 100) if total_market > 0 else 0
        
        # Price positioning
        competitor_prices = [c['avg_price'] for c in competitor_metrics]
        your_price = your_metrics['avg_price']
        
        price_percentile = sum(1 for p in competitor_prices if your_price > p) / len(competitor_prices) * 100
        
        # Quality positioning (based on customer rating)
        competitor_ratings = [c['customer_rating'] for c in competitor_metrics]
        your_rating = your_metrics['customer_rating']
        
        rating_percentile = sum(1 for r in competitor_ratings if your_rating > r) / len(competitor_ratings) * 100
        
        # Determine strategy
        if price_percentile > 70 and rating_percentile > 70:
            position = "Premium Leader"
            strategy = "Maintain premium quality and pricing"
        elif price_percentile < 30 and rating_percentile > 50:
            position = "Value Leader"
            strategy = "Leverage value proposition in marketing"
        elif price_percentile > 50 and rating_percentile > 50:
            position = "Market Leader"
            strategy = "Defend market position, innovate"
        elif price_percentile < 50 and rating_percentile < 50:
            position = "Challenger"
            strategy = "Improve quality or reduce costs"
        else:
            position = "Niche Player"
            strategy = "Focus on differentiation"
        
        return {
            'market_share': round(your_share, 2),
            'competitive_position': position,
            'recommended_strategy': strategy,
            'price_position': f"{round(price_percentile, 1)}th percentile",
            'quality_position': f"{round(rating_percentile, 1)}th percentile",
            'key_competitors': sorted(
                competitor_metrics, 
                key=lambda x: x['revenue'], 
                reverse=True
            )[:3]
        }
    
    def identify_market_gaps(self, market_segments: List[Dict]) -> List[Dict]:
        """Identify underserved market segments"""
        gaps = []
        
        for segment in market_segments:
            demand = segment['demand']
            supply = segment['current_supply']
            
            if demand > supply * 1.3:  # 30% undersupply
                gap_size = demand - supply
                opportunity_score = (gap_size / demand) * 100
                
                gaps.append({
                    'segment': segment['name'],
                    'gap_size': round(gap_size, 2),
                    'opportunity_score': round(opportunity_score, 2),
                    'recommendation': f"High opportunity in {segment['name']}"
                })
        
        return sorted(gaps, key=lambda x: x['opportunity_score'], reverse=True)


class RetailIntelligencePlatform:
    """Main platform integrating all AI engines"""
    
    def __init__(self):
        self.market_engine = MarketIntelligenceEngine()
        self.pricing_engine = PricingIntelligence()
        self.customer_engine = CustomerInsightsEngine()
        self.competitor_analyzer = CompetitorAnalyzer()
        
    def generate_comprehensive_report(self, business_data: Dict) -> Dict:
        """Generate comprehensive business intelligence report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'Comprehensive Retail Intelligence',
            'insights': {}
        }
        
        # Market Analysis
        if 'sales_data' in business_data:
            market_trends = self.market_engine.analyze_market_trends(
                business_data['sales_data']
            )
            demand_forecast = self.market_engine.forecast_demand(
                business_data['sales_data'], 
                periods=30
            )
            
            report['insights']['market_analysis'] = {
                'trends': market_trends,
                'forecast_summary': {
                    'next_30_days_avg': round(demand_forecast['forecasted_sales'].mean(), 2),
                    'growth_potential': round(
                        (demand_forecast['forecasted_sales'].iloc[-1] / 
                         business_data['sales_data']['sales'].iloc[-1] - 1) * 100, 2
                    )
                }
            }
        
        # Pricing Intelligence
        if 'pricing_data' in business_data:
            elasticity = self.pricing_engine.analyze_price_elasticity(
                business_data['pricing_data']
            )
            pricing_optimization = self.pricing_engine.optimize_pricing(
                business_data['current_price'],
                business_data['cost'],
                business_data['demand_forecast'],
                business_data['competitor_prices']
            )
            
            report['insights']['pricing'] = {
                'elasticity': elasticity,
                'optimization': pricing_optimization
            }
        
        # Customer Insights
        if 'customer_data' in business_data:
            segments = self.customer_engine.segment_customers(
                business_data['customer_data']
            )
            
            report['insights']['customers'] = segments
        
        # Competitive Analysis
        if 'competitive_data' in business_data:
            competitive_position = self.competitor_analyzer.analyze_competitive_position(
                business_data['your_metrics'],
                business_data['competitor_metrics']
            )
            
            report['insights']['competition'] = competitive_position
        
        # Executive Summary
        report['executive_summary'] = self._generate_executive_summary(report['insights'])
        
        return report
    
    def _generate_executive_summary(self, insights: Dict) -> Dict:
        """Generate executive summary with key recommendations"""
        summary = {
            'top_opportunities': [],
            'critical_actions': [],
            'performance_indicators': {}
        }
        
        # Extract key opportunities
        if 'market_analysis' in insights:
            trend = insights['market_analysis']['trends']['overall_trend']
            growth = insights['market_analysis']['forecast_summary']['growth_potential']
            
            if growth > 10:
                summary['top_opportunities'].append(
                    f"Strong growth potential: {growth}% forecasted increase"
                )
            
            summary['performance_indicators']['market_trend'] = trend
        
        if 'pricing' in insights:
            elasticity = insights['pricing']['elasticity']['category']
            recommended_price = insights['pricing']['optimization']['recommended_price']
            
            summary['critical_actions'].append(
                f"Consider {insights['pricing']['optimization']['recommended_strategy']} "
                f"strategy at ${recommended_price}"
            )
            
            summary['performance_indicators']['pricing_strategy'] = elasticity
        
        if 'customers' in insights:
            top_segment = max(
                insights['customers']['segment_value'].items(),
                key=lambda x: x[1]
            )
            
            summary['top_opportunities'].append(
                f"Focus on {top_segment[0]} segment (${top_segment[1]:,.2f} value)"
            )
        
        if 'competition' in insights:
            position = insights['competition']['competitive_position']
            summary['performance_indicators']['market_position'] = position
            summary['critical_actions'].append(
                insights['competition']['recommended_strategy']
            )
        
        return summary


# Demo function
def run_demo():
    """Run a demonstration of the platform"""
    
    print("=" * 80)
    print("AI-POWERED RETAIL INTELLIGENCE PLATFORM - DEMO")
    print("=" * 80)
    print()
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    
    # Sales data with trend and seasonality
    trend = np.linspace(1000, 1500, 90)
    seasonality = 200 * np.sin(np.linspace(0, 4*np.pi, 90))
    noise = np.random.normal(0, 50, 90)
    sales = trend + seasonality + noise
    
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    
    # Pricing data
    pricing_data = pd.DataFrame({
        'price': [29.99, 27.99, 25.99, 28.99, 30.99],
        'quantity': [1000, 1150, 1300, 1100, 950]
    })
    
    # Customer data
    customer_data = pd.DataFrame({
        'customer_id': range(1, 101),
        'days_since_purchase': np.random.randint(1, 365, 100),
        'purchase_count': np.random.randint(1, 20, 100),
        'total_spent': np.random.uniform(100, 5000, 100)
    })
    
    # Competitive data
    your_metrics = {
        'revenue': 500000,
        'avg_price': 29.99,
        'customer_rating': 4.5
    }
    
    competitor_metrics = [
        {'name': 'Competitor A', 'revenue': 600000, 'avg_price': 32.99, 'customer_rating': 4.3},
        {'name': 'Competitor B', 'revenue': 450000, 'avg_price': 27.99, 'customer_rating': 4.1},
        {'name': 'Competitor C', 'revenue': 400000, 'avg_price': 25.99, 'customer_rating': 3.9},
    ]
    
    # Create platform
    platform = RetailIntelligencePlatform()
    
    # Generate report
    business_data = {
        'sales_data': sales_data,
        'pricing_data': pricing_data,
        'current_price': 29.99,
        'cost': 18.00,
        'demand_forecast': 1200,
        'competitor_prices': [32.99, 27.99, 25.99, 28.99],
        'customer_data': customer_data,
        'your_metrics': your_metrics,
        'competitor_metrics': competitor_metrics
    }
    
    report = platform.generate_comprehensive_report(business_data)
    
    # Display results
    print("\n📊 EXECUTIVE SUMMARY")
    print("-" * 80)
    for key, value in report['executive_summary']['performance_indicators'].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print("\n🎯 TOP OPPORTUNITIES")
    for opp in report['executive_summary']['top_opportunities']:
        print(f"  • {opp}")
    
    print("\n⚡ CRITICAL ACTIONS")
    for action in report['executive_summary']['critical_actions']:
        print(f"  • {action}")
    
    print("\n📈 MARKET ANALYSIS")
    print("-" * 80)
    market = report['insights']['market_analysis']
    print(f"  Trend: {market['trends']['overall_trend']}")
    print(f"  Growth Rate: {market['trends']['growth_rate']}%")
    print(f"  Volatility: {market['trends']['volatility']}%")
    print(f"  30-Day Forecast Avg: {market['forecast_summary']['next_30_days_avg']:.2f} units")
    
    print("\n💰 PRICING INTELLIGENCE")
    print("-" * 80)
    pricing = report['insights']['pricing']
    print(f"  Price Elasticity: {pricing['elasticity']['elasticity']} ({pricing['elasticity']['category']})")
    print(f"  Recommendation: {pricing['elasticity']['recommendation']}")
    print(f"  Optimal Strategy: {pricing['optimization']['recommended_strategy']}")
    print(f"  Recommended Price: ${pricing['optimization']['recommended_price']}")
    
    print("\n👥 CUSTOMER SEGMENTS")
    print("-" * 80)
    for segment, count in report['insights']['customers']['segments'].items():
        value = report['insights']['customers']['segment_value'].get(segment, 0)
        recommendation = report['insights']['customers']['recommendations'].get(segment, '')
        print(f"  {segment}: {count} customers (${value:,.2f})")
        print(f"    → {recommendation}")
    
    if 'competition' in report['insights']:
        print("\n🏆 COMPETITIVE POSITION")
        print("-" * 80)
        comp = report['insights']['competition']
        print(f"  Market Share: {comp['market_share']}%")
        print(f"  Position: {comp['competitive_position']}")
        print(f"  Strategy: {comp['recommended_strategy']}")
        print(f"  Price Position: {comp['price_position']}")
        print(f"  Quality Position: {comp['quality_position']}")
    
    print("\n" + "=" * 80)
    print("Report generated successfully!")
    print("=" * 80)
    
    return report


if __name__ == "__main__":
    report = run_demo()
    print("\n✅ Platform demonstration complete!")
    print("📁 Full report data available in the 'report' variable")
