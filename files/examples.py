"""
Practical Example: Using the Retail Intelligence Platform
=========================================================
This script demonstrates how to use the platform with your own data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from retail_intelligence_platform import RetailIntelligencePlatform
from advanced_analytics import (
    MLPricingOptimizer, 
    AnomalyDetector, 
    ChurnPredictor,
    InventoryOptimizer
)


def example_1_basic_market_analysis():
    """Example 1: Analyze your sales trends and forecast demand"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Market Analysis")
    print("="*80)
    
    # Load your sales data (replace with your actual data)
    # For this example, we'll create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Simulate realistic sales data with trend, seasonality, and noise
    trend = np.linspace(800, 1500, len(dates))
    seasonality = 300 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
    noise = np.random.normal(0, 80, len(dates))
    sales = trend + seasonality + noise
    
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    
    # Initialize platform
    platform = RetailIntelligencePlatform()
    
    # Analyze trends
    trends = platform.market_engine.analyze_market_trends(sales_data)
    
    print("\n📊 Market Trends Analysis:")
    print(f"  Overall Trend: {trends['overall_trend']}")
    print(f"  Growth Rate: {trends['growth_rate']}% annually")
    print(f"  Volatility: {trends['volatility']}%")
    
    if 'peak_month' in trends['seasonality']:
        print(f"  Peak Season: Month {trends['seasonality']['peak_month']}")
        print(f"  Low Season: Month {trends['seasonality']['low_month']}")
    
    # Forecast next 30 days
    forecast = platform.market_engine.forecast_demand(sales_data, periods=30)
    
    print("\n📈 30-Day Demand Forecast:")
    print(f"  Average Forecasted Sales: {forecast['forecasted_sales'].mean():.0f} units/day")
    print(f"  Peak Forecast: {forecast['forecasted_sales'].max():.0f} units")
    print(f"  Low Forecast: {forecast['forecasted_sales'].min():.0f} units")
    
    # Save forecast to CSV
    forecast.to_csv('/home/claude/demand_forecast.csv', index=False)
    print("\n✅ Forecast saved to: demand_forecast.csv")
    
    return forecast


def example_2_pricing_optimization():
    """Example 2: Optimize your product pricing"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Pricing Optimization")
    print("="*80)
    
    # Historical pricing experiments
    pricing_data = pd.DataFrame({
        'price': [24.99, 26.99, 28.99, 30.99, 32.99, 34.99],
        'quantity': [1500, 1350, 1200, 1050, 900, 750]
    })
    
    platform = RetailIntelligencePlatform()
    
    # Analyze price elasticity
    elasticity = platform.pricing_engine.analyze_price_elasticity(pricing_data)
    
    print("\n💰 Price Elasticity Analysis:")
    print(f"  Elasticity Coefficient: {elasticity['elasticity']}")
    print(f"  Category: {elasticity['category']}")
    print(f"  Recommendation: {elasticity['recommendation']}")
    
    # Get pricing recommendations
    current_price = 29.99
    cost = 18.50
    demand_forecast = 1200
    competitor_prices = [32.99, 27.99, 26.49, 31.99, 28.99]
    
    optimization = platform.pricing_engine.optimize_pricing(
        current_price, cost, demand_forecast, competitor_prices
    )
    
    print("\n🎯 Pricing Strategy Recommendations:")
    print(f"  Current Price: ${optimization['current_price']}")
    print(f"  Recommended Strategy: {optimization['recommended_strategy']}")
    print(f"  Recommended Price: ${optimization['recommended_price']}")
    print(f"  Market Position: {optimization['market_position']}")
    
    print("\n📊 All Strategy Comparison:")
    for strategy, metrics in optimization['all_strategies'].items():
        print(f"\n  {strategy.upper()}:")
        print(f"    Price: ${metrics['price']}")
        print(f"    Est. Revenue: ${metrics['estimated_revenue']:,.2f}")
        print(f"    Profit Margin: {metrics['profit_margin']:.1f}%")
    
    # Advanced ML optimization
    ml_optimizer = MLPricingOptimizer()
    ml_optimizer.train_price_demand_model(pricing_data)
    optimal = ml_optimizer.find_optimal_price(cost)
    
    print("\n🤖 ML-Optimized Pricing:")
    print(f"  Revenue-Maximizing Price: ${optimal['revenue_maximizing_price']}")
    print(f"  Expected Revenue: ${optimal['max_revenue']:,.2f}")
    print(f"  Profit-Maximizing Price: ${optimal['profit_maximizing_price']}")
    print(f"  Expected Profit: ${optimal['max_profit']:,.2f}")


def example_3_customer_segmentation():
    """Example 3: Segment customers and reduce churn"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Customer Segmentation & Churn Prevention")
    print("="*80)
    
    # Create sample customer data
    np.random.seed(42)
    n_customers = 200
    
    customer_data = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'days_since_purchase': np.random.randint(1, 365, n_customers),
        'purchase_count': np.random.randint(1, 50, n_customers),
        'total_spent': np.random.uniform(100, 10000, n_customers)
    })
    
    platform = RetailIntelligencePlatform()
    
    # Segment customers
    segments = platform.customer_engine.segment_customers(customer_data)
    
    print("\n👥 Customer Segmentation Results:")
    total_customers = sum(segments['segments'].values())
    total_value = sum(segments['segment_value'].values())
    
    for segment in sorted(segments['segments'].keys(), 
                         key=lambda x: segments['segment_value'].get(x, 0), 
                         reverse=True):
        count = segments['segments'][segment]
        value = segments['segment_value'].get(segment, 0)
        pct = (count / total_customers) * 100
        value_pct = (value / total_value) * 100
        
        print(f"\n  {segment}:")
        print(f"    Customers: {count} ({pct:.1f}%)")
        print(f"    Total Value: ${value:,.2f} ({value_pct:.1f}%)")
        print(f"    Action: {segments['recommendations'][segment]}")
    
    # Churn prediction
    predictor = ChurnPredictor()
    churn_analysis = predictor.predict_churn_risk(customer_data)
    
    print("\n⚠️  High Churn Risk Customers:")
    high_risk = churn_analysis[churn_analysis['churn_risk_level'] == 'High'].head(10)
    
    for _, customer in high_risk.iterrows():
        print(f"\n  Customer #{customer['customer_id']}:")
        print(f"    Churn Risk: {customer['churn_risk_score']:.1f}%")
        print(f"    Days Since Purchase: {customer['days_since_purchase']}")
        print(f"    Total Spent: ${customer['total_spent']:.2f}")
        print(f"    Action: {customer['retention_action']}")
    
    # Save churn analysis
    churn_analysis.to_csv('/home/claude/churn_analysis.csv', index=False)
    print("\n✅ Churn analysis saved to: churn_analysis.csv")


def example_4_inventory_optimization():
    """Example 4: Optimize inventory levels"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Inventory Optimization")
    print("="*80)
    
    # Historical demand data
    demand_history = [
        105, 98, 112, 95, 108, 102, 97, 110, 100, 106,
        99, 115, 103, 96, 109, 101, 107, 98, 104, 111,
        97, 105, 100, 113, 99, 106, 102, 108, 95, 110
    ]
    
    optimizer = InventoryOptimizer()
    
    # Calculate optimal inventory levels
    inventory_plan = optimizer.calculate_optimal_stock(
        demand_history,
        lead_time_days=7,
        service_level=0.95
    )
    
    print("\n📦 Optimal Inventory Plan:")
    print(f"  Average Daily Demand: {inventory_plan['average_daily_demand']:.1f} units")
    print(f"  Demand Variability (Std Dev): {inventory_plan['demand_variability']:.1f}")
    print(f"\n  Safety Stock: {inventory_plan['safety_stock']:.0f} units")
    print(f"  Reorder Point: {inventory_plan['reorder_point']:.0f} units")
    print(f"  Economic Order Quantity: {inventory_plan['economic_order_quantity']:.0f} units")
    print(f"\n  Max Inventory Level: {inventory_plan['max_inventory_level']:.0f} units")
    print(f"  Average Inventory: {inventory_plan['average_inventory_level']:.0f} units")
    print(f"\n  Service Level: {inventory_plan['service_level']:.0f}%")
    print(f"  Stockout Risk: {inventory_plan['stockout_risk']:.2f}%")
    
    print("\n💡 Inventory Management Tips:")
    print(f"  • Order {inventory_plan['economic_order_quantity']:.0f} units each time")
    print(f"  • Reorder when stock reaches {inventory_plan['reorder_point']:.0f} units")
    print(f"  • Maintain {inventory_plan['safety_stock']:.0f} units as safety buffer")
    print(f"  • This achieves {inventory_plan['service_level']:.0f}% service level")


def example_5_comprehensive_report():
    """Example 5: Generate comprehensive business intelligence report"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Comprehensive Business Intelligence Report")
    print("="*80)
    
    # Prepare all business data
    np.random.seed(42)
    
    # Sales data
    dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
    trend = np.linspace(1000, 1800, len(dates))
    seasonality = 300 * np.sin(np.linspace(0, 6*np.pi, len(dates)))
    sales = trend + seasonality + np.random.normal(0, 100, len(dates))
    
    sales_data = pd.DataFrame({'date': dates, 'sales': sales})
    
    # Pricing data
    pricing_data = pd.DataFrame({
        'price': [25, 27, 29, 31, 33],
        'quantity': [1400, 1300, 1150, 1000, 850]
    })
    
    # Customer data
    customer_data = pd.DataFrame({
        'customer_id': range(1, 301),
        'days_since_purchase': np.random.randint(1, 365, 300),
        'purchase_count': np.random.randint(1, 40, 300),
        'total_spent': np.random.uniform(200, 12000, 300)
    })
    
    # Competitive data
    your_metrics = {
        'revenue': 950000,
        'avg_price': 29.99,
        'customer_rating': 4.5
    }
    
    competitor_metrics = [
        {'name': 'Leader Co', 'revenue': 1500000, 'avg_price': 35.99, 'customer_rating': 4.6},
        {'name': 'Value Inc', 'revenue': 800000, 'avg_price': 24.99, 'customer_rating': 4.1},
        {'name': 'Premium Ltd', 'revenue': 650000, 'avg_price': 42.99, 'customer_rating': 4.8},
    ]
    
    # Compile all data
    business_data = {
        'sales_data': sales_data,
        'pricing_data': pricing_data,
        'current_price': 29.99,
        'cost': 19.00,
        'demand_forecast': 1400,
        'competitor_prices': [35.99, 24.99, 42.99, 28.99, 31.49],
        'customer_data': customer_data,
        'your_metrics': your_metrics,
        'competitor_metrics': competitor_metrics
    }
    
    # Generate comprehensive report
    platform = RetailIntelligencePlatform()
    report = platform.generate_comprehensive_report(business_data)
    
    # Display executive summary
    print("\n📊 EXECUTIVE SUMMARY")
    print("-" * 80)
    
    summary = report['executive_summary']
    
    print("\n🎯 Key Performance Indicators:")
    for kpi, value in summary['performance_indicators'].items():
        print(f"  • {kpi.replace('_', ' ').title()}: {value}")
    
    print("\n💡 Top Opportunities:")
    for i, opp in enumerate(summary['top_opportunities'], 1):
        print(f"  {i}. {opp}")
    
    print("\n⚡ Critical Actions:")
    for i, action in enumerate(summary['critical_actions'], 1):
        print(f"  {i}. {action}")
    
    # Save full report
    import json
    
    # Convert report to JSON-friendly format
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        return obj
    
    serializable_report = make_serializable(report)
    
    with open('/home/claude/comprehensive_report.json', 'w') as f:
        json.dump(serializable_report, f, indent=2, default=str)
    
    print("\n✅ Full report saved to: comprehensive_report.json")
    
    return report


def main():
    """Run all examples"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "RETAIL INTELLIGENCE PLATFORM" + " "*30 + "║")
    print("║" + " "*25 + "Practical Examples" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    # Run examples
    print("\n\nRunning 5 practical examples demonstrating platform capabilities...")
    
    example_1_basic_market_analysis()
    example_2_pricing_optimization()
    example_3_customer_segmentation()
    example_4_inventory_optimization()
    example_5_comprehensive_report()
    
    print("\n\n" + "="*80)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print("  • demand_forecast.csv - 30-day sales forecast")
    print("  • churn_analysis.csv - Customer churn risk analysis")
    print("  • comprehensive_report.json - Full business intelligence report")
    
    print("\n💡 Next Steps:")
    print("  1. Replace sample data with your actual business data")
    print("  2. Customize thresholds and parameters for your industry")
    print("  3. Integrate with your existing systems via the API")
    print("  4. Set up automated daily/weekly reports")
    print("  5. Train your team on interpreting the insights")
    
    print("\n📚 For more information:")
    print("  • Read README.md for detailed documentation")
    print("  • Check advanced_analytics.py for ML features")
    print("  • Run app.py to launch the web dashboard")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
