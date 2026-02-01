"""
Interactive Web Dashboard for Retail Intelligence Platform
===========================================================
Real-time visualization and insights
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from retail_intelligence_platform import RetailIntelligencePlatform

app = Flask(__name__)
platform = RetailIntelligencePlatform()

# Sample data generator
def generate_sample_data():
    """Generate realistic sample data for demonstration"""
    np.random.seed(42)
    
    # 90 days of sales data
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), periods=90, freq='D')
    trend = np.linspace(1000, 1500, 90)
    seasonality = 200 * np.sin(np.linspace(0, 4*np.pi, 90))
    noise = np.random.normal(0, 50, 90)
    sales = trend + seasonality + noise
    
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    
    # Pricing experiments
    pricing_data = pd.DataFrame({
        'price': [29.99, 27.99, 25.99, 28.99, 30.99, 26.99, 29.49],
        'quantity': [1000, 1150, 1300, 1100, 950, 1280, 1050]
    })
    
    # Customer segments (500 customers)
    customer_data = pd.DataFrame({
        'customer_id': range(1, 501),
        'days_since_purchase': np.random.randint(1, 365, 500),
        'purchase_count': np.random.randint(1, 30, 500),
        'total_spent': np.random.uniform(50, 8000, 500)
    })
    
    # Your business metrics
    your_metrics = {
        'revenue': 750000,
        'avg_price': 29.99,
        'customer_rating': 4.5
    }
    
    # Competitor data
    competitor_metrics = [
        {'name': 'Market Leader Inc', 'revenue': 1200000, 'avg_price': 34.99, 'customer_rating': 4.6},
        {'name': 'Value Retailer Co', 'revenue': 650000, 'avg_price': 24.99, 'customer_rating': 4.0},
        {'name': 'Premium Brands Ltd', 'revenue': 550000, 'avg_price': 39.99, 'customer_rating': 4.7},
        {'name': 'Quick Commerce', 'revenue': 480000, 'avg_price': 27.99, 'customer_rating': 3.9},
    ]
    
    return {
        'sales_data': sales_data,
        'pricing_data': pricing_data,
        'current_price': 29.99,
        'cost': 18.50,
        'demand_forecast': 1250,
        'competitor_prices': [34.99, 24.99, 39.99, 27.99, 29.99],
        'customer_data': customer_data,
        'your_metrics': your_metrics,
        'competitor_metrics': competitor_metrics
    }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/analysis')
def get_analysis():
    """API endpoint for full analysis"""
    business_data = generate_sample_data()
    report = platform.generate_comprehensive_report(business_data)
    
    # Convert numpy/pandas types to JSON-serializable formats
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    return jsonify(convert_to_serializable(report))

@app.route('/api/forecast')
def get_forecast():
    """Get demand forecast"""
    business_data = generate_sample_data()
    forecast = platform.market_engine.forecast_demand(business_data['sales_data'], periods=30)
    
    return jsonify({
        'periods': forecast['period'].tolist(),
        'forecast': forecast['forecasted_sales'].tolist(),
        'lower_bound': forecast['confidence_lower'].tolist(),
        'upper_bound': forecast['confidence_upper'].tolist()
    })

@app.route('/api/pricing')
def get_pricing_optimization():
    """Get pricing recommendations"""
    business_data = generate_sample_data()
    
    elasticity = platform.pricing_engine.analyze_price_elasticity(business_data['pricing_data'])
    optimization = platform.pricing_engine.optimize_pricing(
        business_data['current_price'],
        business_data['cost'],
        business_data['demand_forecast'],
        business_data['competitor_prices']
    )
    
    return jsonify({
        'elasticity': elasticity,
        'optimization': optimization
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 RETAIL INTELLIGENCE PLATFORM - WEB DASHBOARD")
    print("="*80)
    print("\nStarting server...")
    print("Dashboard will be available at: http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  • GET /api/analysis - Full business analysis")
    print("  • GET /api/forecast - Demand forecasting")
    print("  • GET /api/pricing - Pricing optimization")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5000)
