"""
Advanced Analytics Module
=========================
Machine learning models for deeper insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json


class MLPricingOptimizer:
    """Machine Learning-based pricing optimization"""
    
    def __init__(self):
        self.price_history = []
        self.demand_history = []
        
    def train_price_demand_model(self, historical_data: pd.DataFrame):
        """Train a simple price-demand relationship model"""
        self.price_history = historical_data['price'].values
        self.demand_history = historical_data['quantity'].values
        
        # Polynomial regression for non-linear relationships
        self.coefficients = np.polyfit(self.price_history, self.demand_history, 2)
        
    def predict_demand(self, price: float) -> float:
        """Predict demand at given price point"""
        if not hasattr(self, 'coefficients'):
            return 0
            
        # Polynomial prediction
        demand = np.polyval(self.coefficients, price)
        return max(0, demand)  # Ensure non-negative
    
    def find_optimal_price(self, cost: float, min_price: float = None, 
                          max_price: float = None) -> Dict:
        """Find revenue-maximizing price point"""
        if min_price is None:
            min_price = cost * 1.1  # At least 10% margin
        if max_price is None:
            max_price = cost * 3.0  # Up to 200% margin
            
        # Test price points
        test_prices = np.linspace(min_price, max_price, 100)
        revenues = []
        profits = []
        
        for price in test_prices:
            demand = self.predict_demand(price)
            revenue = price * demand
            profit = (price - cost) * demand
            revenues.append(revenue)
            profits.append(profit)
        
        # Find optimal points
        max_revenue_idx = np.argmax(revenues)
        max_profit_idx = np.argmax(profits)
        
        return {
            'revenue_maximizing_price': round(test_prices[max_revenue_idx], 2),
            'max_revenue': round(revenues[max_revenue_idx], 2),
            'profit_maximizing_price': round(test_prices[max_profit_idx], 2),
            'max_profit': round(profits[max_profit_idx], 2),
            'demand_at_optimal': round(self.predict_demand(test_prices[max_profit_idx]), 0)
        }


class AnomalyDetector:
    """Detect anomalies in sales patterns"""
    
    def detect_anomalies(self, time_series: pd.Series, 
                        threshold: float = 2.5) -> Dict:
        """Detect anomalies using statistical methods"""
        
        # Calculate rolling statistics
        rolling_mean = time_series.rolling(window=7, center=True).mean()
        rolling_std = time_series.rolling(window=7, center=True).std()
        
        # Z-score method
        z_scores = np.abs((time_series - rolling_mean) / rolling_std)
        anomalies = z_scores > threshold
        
        anomaly_indices = np.where(anomalies)[0]
        anomaly_dates = time_series.index[anomaly_indices].tolist() if hasattr(time_series.index, 'tolist') else anomaly_indices.tolist()
        anomaly_values = time_series.iloc[anomaly_indices].tolist()
        
        # Classify anomalies
        positive_anomalies = sum(1 for i in anomaly_indices if time_series.iloc[i] > rolling_mean.iloc[i])
        negative_anomalies = len(anomaly_indices) - positive_anomalies
        
        return {
            'total_anomalies': len(anomaly_indices),
            'positive_anomalies': positive_anomalies,  # Unusually high
            'negative_anomalies': negative_anomalies,  # Unusually low
            'anomaly_dates': [str(d) for d in anomaly_dates[:10]],  # Top 10
            'anomaly_values': [float(v) for v in anomaly_values[:10]],
            'anomaly_rate': round(len(anomaly_indices) / len(time_series) * 100, 2)
        }


class ProductRecommendationEngine:
    """Product recommendation and bundling suggestions"""
    
    def analyze_product_affinity(self, transaction_data: pd.DataFrame) -> Dict:
        """Analyze which products are frequently bought together"""
        
        # Simple market basket analysis
        # Assume transaction_data has 'transaction_id' and 'product_id'
        
        # Get unique products
        products = transaction_data['product_id'].unique()
        
        # Calculate co-occurrence matrix
        co_occurrence = {}
        
        for txn in transaction_data['transaction_id'].unique():
            items = transaction_data[
                transaction_data['transaction_id'] == txn
            ]['product_id'].tolist()
            
            # Record co-occurrences
            for i, item1 in enumerate(items):
                for item2 in items[i+1:]:
                    key = tuple(sorted([item1, item2]))
                    co_occurrence[key] = co_occurrence.get(key, 0) + 1
        
        # Find top associations
        top_associations = sorted(
            co_occurrence.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Calculate lift (strength of association)
        total_transactions = transaction_data['transaction_id'].nunique()
        product_counts = transaction_data['product_id'].value_counts()
        
        associations_with_lift = []
        for (prod1, prod2), count in top_associations:
            support = count / total_transactions
            confidence = count / product_counts.get(prod1, 1)
            
            expected = (product_counts.get(prod1, 0) / total_transactions) * \
                      (product_counts.get(prod2, 0) / total_transactions)
            lift = support / expected if expected > 0 else 0
            
            associations_with_lift.append({
                'product_1': prod1,
                'product_2': prod2,
                'frequency': count,
                'support': round(support, 4),
                'confidence': round(confidence, 4),
                'lift': round(lift, 2)
            })
        
        return {
            'top_product_pairs': associations_with_lift,
            'bundle_recommendations': self._generate_bundle_recommendations(associations_with_lift)
        }
    
    def _generate_bundle_recommendations(self, associations: List[Dict]) -> List[str]:
        """Generate actionable bundling recommendations"""
        recommendations = []
        
        for assoc in associations[:5]:
            if assoc['lift'] > 1.5:  # Strong association
                recommendations.append(
                    f"Create bundle: {assoc['product_1']} + {assoc['product_2']} "
                    f"(Lift: {assoc['lift']}x more likely together)"
                )
        
        return recommendations


class ChurnPredictor:
    """Predict customer churn risk"""
    
    def predict_churn_risk(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate churn risk score for each customer"""
        
        customer_data = customer_data.copy()
        
        # Calculate risk factors
        # 1. Recency risk (days since last purchase)
        max_days = customer_data['days_since_purchase'].max()
        recency_risk = customer_data['days_since_purchase'] / max_days
        
        # 2. Frequency risk (inverse of purchase frequency)
        max_frequency = customer_data['purchase_count'].max()
        frequency_risk = 1 - (customer_data['purchase_count'] / max_frequency)
        
        # 3. Monetary risk (inverse of spending)
        max_spent = customer_data['total_spent'].max()
        monetary_risk = 1 - (customer_data['total_spent'] / max_spent)
        
        # Combined risk score (weighted average)
        churn_score = (
            0.5 * recency_risk + 
            0.3 * frequency_risk + 
            0.2 * monetary_risk
        ) * 100
        
        customer_data['churn_risk_score'] = churn_score.round(2)
        customer_data['churn_risk_level'] = pd.cut(
            churn_score,
            bins=[0, 30, 60, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        # Generate retention recommendations
        customer_data['retention_action'] = customer_data['churn_risk_level'].map({
            'Low': 'Monitor - Continue regular engagement',
            'Medium': 'Engage - Send personalized offers',
            'High': 'Urgent - Immediate win-back campaign'
        })
        
        return customer_data


class InventoryOptimizer:
    """Optimize inventory levels"""
    
    def calculate_optimal_stock(self, historical_demand: List[float],
                               lead_time_days: int = 7,
                               service_level: float = 0.95) -> Dict:
        """Calculate optimal inventory levels"""
        
        demand_array = np.array(historical_demand)
        
        # Calculate demand statistics
        avg_daily_demand = demand_array.mean()
        demand_std = demand_array.std()
        
        # Safety stock calculation (assumes normal distribution)
        z_score = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}.get(service_level, 1.65)
        safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
        
        # Reorder point
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        # Economic Order Quantity (EOQ) - simplified
        annual_demand = avg_daily_demand * 365
        ordering_cost = 100  # Assumed
        holding_cost = 5  # Assumed per unit per year
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        
        # Calculate inventory metrics
        max_inventory = reorder_point + eoq
        avg_inventory = (safety_stock + eoq/2)
        
        return {
            'average_daily_demand': round(avg_daily_demand, 2),
            'demand_variability': round(demand_std, 2),
            'safety_stock': round(safety_stock, 0),
            'reorder_point': round(reorder_point, 0),
            'economic_order_quantity': round(eoq, 0),
            'max_inventory_level': round(max_inventory, 0),
            'average_inventory_level': round(avg_inventory, 0),
            'service_level': service_level * 100,
            'stockout_risk': round((1 - service_level) * 100, 2)
        }
    
    def detect_slow_moving_items(self, product_sales: pd.DataFrame,
                                 threshold_days: int = 90) -> List[Dict]:
        """Identify slow-moving inventory"""
        
        slow_movers = []
        
        for _, row in product_sales.iterrows():
            if row['days_since_sale'] > threshold_days:
                inventory_value = row['units_in_stock'] * row['cost_per_unit']
                
                slow_movers.append({
                    'product_id': row['product_id'],
                    'product_name': row['product_name'],
                    'days_since_sale': row['days_since_sale'],
                    'units_in_stock': row['units_in_stock'],
                    'inventory_value': round(inventory_value, 2),
                    'recommendation': self._get_slow_mover_action(row['days_since_sale'])
                })
        
        return sorted(slow_movers, key=lambda x: x['inventory_value'], reverse=True)
    
    def _get_slow_mover_action(self, days_since_sale: int) -> str:
        """Recommend action for slow-moving items"""
        if days_since_sale > 180:
            return "Liquidate - Deep discount or donation"
        elif days_since_sale > 120:
            return "Promote - Run clearance sale"
        else:
            return "Monitor - Reduce future orders"


class SeasonalityAnalyzer:
    """Advanced seasonality detection and forecasting"""
    
    def decompose_time_series(self, time_series: pd.Series) -> Dict:
        """Decompose time series into trend, seasonal, and residual components"""
        
        # Simple moving average for trend
        window = min(7, len(time_series) // 3)
        trend = time_series.rolling(window=window, center=True).mean()
        
        # Detrend
        detrended = time_series - trend
        
        # Simple seasonal pattern (weekly if enough data)
        if len(time_series) >= 14:
            # Calculate average for each day of week
            seasonal_pattern = detrended.groupby(
                detrended.index.dayofweek if hasattr(detrended.index, 'dayofweek') 
                else detrended.index % 7
            ).mean()
            
            # Expand seasonal pattern to full series
            seasonal = pd.Series(
                [seasonal_pattern.get(i % 7, 0) for i in range(len(time_series))],
                index=time_series.index
            )
        else:
            seasonal = pd.Series(0, index=time_series.index)
        
        # Residual (random component)
        residual = time_series - trend - seasonal
        
        return {
            'trend_strength': round(trend.std() / time_series.std() * 100, 2) if time_series.std() > 0 else 0,
            'seasonal_strength': round(seasonal.std() / time_series.std() * 100, 2) if time_series.std() > 0 else 0,
            'residual_strength': round(residual.std() / time_series.std() * 100, 2) if time_series.std() > 0 else 0,
            'dominant_pattern': self._identify_dominant_pattern(trend, seasonal, residual)
        }
    
    def _identify_dominant_pattern(self, trend, seasonal, residual) -> str:
        """Identify the strongest pattern in the data"""
        strengths = {
            'Trending': trend.std(),
            'Seasonal': seasonal.std(),
            'Random': residual.std()
        }
        
        dominant = max(strengths.items(), key=lambda x: x[1])
        return dominant[0]


# Demo function for advanced analytics
def demo_advanced_analytics():
    """Demonstrate advanced analytics capabilities"""
    
    print("\n" + "="*80)
    print("ADVANCED ANALYTICS MODULE - DEMONSTRATION")
    print("="*80)
    
    # 1. ML Pricing Optimization
    print("\n🤖 ML PRICING OPTIMIZATION")
    print("-"*80)
    
    pricing_optimizer = MLPricingOptimizer()
    
    # Sample pricing experiments
    price_experiment_data = pd.DataFrame({
        'price': [20, 22, 24, 26, 28, 30, 32, 34],
        'quantity': [1500, 1400, 1250, 1100, 950, 800, 650, 500]
    })
    
    pricing_optimizer.train_price_demand_model(price_experiment_data)
    optimal_prices = pricing_optimizer.find_optimal_price(cost=15.00)
    
    print(f"Revenue-maximizing price: ${optimal_prices['revenue_maximizing_price']}")
    print(f"Expected revenue: ${optimal_prices['max_revenue']:,.2f}")
    print(f"Profit-maximizing price: ${optimal_prices['profit_maximizing_price']}")
    print(f"Expected profit: ${optimal_prices['max_profit']:,.2f}")
    
    # 2. Anomaly Detection
    print("\n🔍 ANOMALY DETECTION")
    print("-"*80)
    
    detector = AnomalyDetector()
    
    # Generate sample time series with anomalies
    np.random.seed(42)
    normal_sales = np.random.normal(1000, 100, 60)
    normal_sales[10] = 1500  # Spike
    normal_sales[25] = 400   # Drop
    normal_sales[45] = 1600  # Spike
    
    sales_series = pd.Series(normal_sales)
    anomalies = detector.detect_anomalies(sales_series)
    
    print(f"Total anomalies detected: {anomalies['total_anomalies']}")
    print(f"Positive anomalies (spikes): {anomalies['positive_anomalies']}")
    print(f"Negative anomalies (drops): {anomalies['negative_anomalies']}")
    print(f"Anomaly rate: {anomalies['anomaly_rate']}%")
    
    # 3. Churn Prediction
    print("\n⚠️  CUSTOMER CHURN PREDICTION")
    print("-"*80)
    
    predictor = ChurnPredictor()
    
    # Sample customer data
    customer_sample = pd.DataFrame({
        'customer_id': range(1, 11),
        'days_since_purchase': [5, 15, 45, 120, 180, 250, 10, 30, 90, 200],
        'purchase_count': [15, 12, 8, 4, 2, 1, 20, 10, 6, 3],
        'total_spent': [5000, 4000, 2500, 1000, 500, 200, 6000, 3500, 2000, 800]
    })
    
    churn_analysis = predictor.predict_churn_risk(customer_sample)
    
    print("\nTop 5 At-Risk Customers:")
    high_risk = churn_analysis.nlargest(5, 'churn_risk_score')
    for _, customer in high_risk.iterrows():
        print(f"  Customer {customer['customer_id']}: "
              f"{customer['churn_risk_score']:.1f}% risk - "
              f"{customer['retention_action']}")
    
    # 4. Inventory Optimization
    print("\n📦 INVENTORY OPTIMIZATION")
    print("-"*80)
    
    optimizer = InventoryOptimizer()
    
    # Sample demand history
    demand_history = [95, 105, 98, 110, 92, 108, 97, 103, 99, 107, 
                     101, 96, 104, 100, 98, 106, 94, 102, 99, 105]
    
    inventory_plan = optimizer.calculate_optimal_stock(
        demand_history, 
        lead_time_days=7, 
        service_level=0.95
    )
    
    print(f"Average daily demand: {inventory_plan['average_daily_demand']} units")
    print(f"Safety stock required: {inventory_plan['safety_stock']} units")
    print(f"Reorder point: {inventory_plan['reorder_point']} units")
    print(f"Economic order quantity: {inventory_plan['economic_order_quantity']} units")
    print(f"Service level: {inventory_plan['service_level']}%")
    print(f"Stockout risk: {inventory_plan['stockout_risk']}%")
    
    print("\n" + "="*80)
    print("✅ Advanced analytics demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    demo_advanced_analytics()
