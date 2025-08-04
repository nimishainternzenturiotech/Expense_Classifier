#!/usr/bin/env python3
"""
Complete Expense Classifier System with Streamlit Interface
This system automatically categorizes expenses using machine learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import re

# Set page config
st.set_page_config(
    page_title=" Expense Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ExpenseClassifier:
    def __init__(self):
        self.categories = [
            "Food & Dining", "Transportation", "Housing & Utilities",
            "Healthcare & Medical", "Shopping & Retail", "Entertainment & Recreation",
            "Education & Learning", "Business & Professional", "Personal Care",
            "Savings & Investment"
        ]
        
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def generate_sample_data(self, n_samples=2000):
        """Generate synthetic expense data for training"""
        
        # Sample descriptions for each category
        sample_data = {
            "Food & Dining": [
                "McDonald's dinner", "Starbucks coffee", "Grocery shopping at Walmart",
                "Pizza delivery", "Restaurant lunch", "Food truck tacos", "Subway sandwich",
                "Ice cream shop", "Fast food burger", "Coffee shop latte", "Bakery pastries",
                "Sushi restaurant", "Italian restaurant", "Chinese takeout", "Breakfast cafe"
            ],
            "Transportation": [
                "Gas station fill-up", "Uber ride", "Bus ticket", "Taxi fare",
                "Car maintenance", "Oil change", "Parking fee", "Subway pass",
                "Flight tickets", "Train ticket", "Car rental", "Auto insurance",
                "Vehicle registration", "Toll road fee", "Car wash"
            ],
            "Housing & Utilities": [
                "Rent payment", "Electricity bill", "Water bill", "Internet service",
                "Phone bill", "Home insurance", "Property tax", "Garbage collection",
                "Home repairs", "Mortgage payment", "HOA fees", "Cable TV",
                "Heating bill", "Security system", "Lawn care"
            ],
            "Healthcare & Medical": [
                "Doctor visit", "Pharmacy prescription", "Dental checkup", "Eye exam",
                "Hospital bill", "Health insurance", "Medical test", "Therapy session",
                "Veterinary care", "Medical supplies", "Urgent care", "Specialist consultation",
                "Physical therapy", "Mental health counseling", "Medical equipment"
            ],
            "Shopping & Retail": [
                "Clothing store", "Electronics purchase", "Home goods", "Online shopping",
                "Department store", "Bookstore", "Hardware store", "Sporting goods",
                "Jewelry store", "Furniture purchase", "Appliance store", "Toy store",
                "Pet supplies", "Office supplies", "Garden center"
            ],
            "Entertainment & Recreation": [
                "Movie theater", "Concert tickets", "Gym membership", "Streaming service",
                "Video games", "Sports event", "Theme park", "Museum visit",
                "Bowling alley", "Mini golf", "Arcade games", "Theater show",
                "Music festival", "Comedy club", "Art class"
            ],
            "Education & Learning": [
                "Tuition payment", "Textbooks", "Online course", "School supplies",
                "Professional training", "Workshop fee", "Certification exam", "Language class",
                "Tutoring session", "Educational software", "Student loan", "School lunch",
                "Laboratory fee", "Library fine", "Academic conference"
            ],
            "Business & Professional": [
                "Business lunch", "Conference registration", "Professional software", "Office rent",
                "Business travel", "Networking event", "Professional membership", "Client dinner",
                "Business cards", "Marketing materials", "Legal services", "Accounting fees",
                "Business insurance", "Equipment lease", "Consulting services"
            ],
            "Personal Care": [
                "Hair salon", "Spa treatment", "Skincare products", "Makeup purchase",
                "Nail salon", "Barbershop", "Massage therapy", "Personal trainer",
                "Beauty products", "Grooming supplies", "Wellness retreat", "Fitness class",
                "Yoga session", "Meditation app", "Health supplements"
            ],
            "Savings & Investment": [
                "Savings account deposit", "Stock purchase", "Mutual fund", "Retirement contribution",
                "Investment advisor fee", "Brokerage fee", "CD deposit", "Bond purchase",
                "Real estate investment", "Cryptocurrency", "Emergency fund", "College fund",
                "Investment research", "Financial planning", "Portfolio management"
            ]
        }
        
        data = []
        for _ in range(n_samples):
            category = random.choice(self.categories)
            description = random.choice(sample_data[category])
            amount = self._generate_realistic_amount(category)
            date = self._generate_random_date()
            
            data.append({
                'date': date,
                'description': description,
                'amount': amount,
                'category': category
            })
        
        return pd.DataFrame(data)
    
    def _generate_realistic_amount(self, category):
        """Generate realistic amounts based on category"""
        amount_ranges = {
            "Food & Dining": (5, 150),
            "Transportation": (10, 500),
            "Housing & Utilities": (50, 2500),
            "Healthcare & Medical": (20, 1000),
            "Shopping & Retail": (15, 800),
            "Entertainment & Recreation": (10, 300),
            "Education & Learning": (25, 5000),
            "Business & Professional": (30, 2000),
            "Personal Care": (15, 400),
            "Savings & Investment": (100, 10000)
        }
        
        min_amount, max_amount = amount_ranges[category]
        return round(random.uniform(min_amount, max_amount), 2)
    
    def _generate_random_date(self):
        """Generate random date within the last year"""
        start_date = datetime.now() - timedelta(days=365)
        random_days = random.randint(0, 365)
        return start_date + timedelta(days=random_days)
    
    def train_model(self, data):
        """Train the classification model"""
        # Prepare features
        X = self.vectorizer.fit_transform(data['description'])
        y = self.label_encoder.fit_transform(data['category'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Get accuracy
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def predict_category(self, description):
        """Predict category for a single expense description"""
        if not self.is_trained:
            return None, 0.0
        
        X = self.vectorizer.transform([description])
        prediction = self.model.predict(X)[0]
        probability = max(self.model.predict_proba(X)[0])
        
        category = self.label_encoder.inverse_transform([prediction])[0]
        return category, probability
    
    def predict_batch(self, descriptions):
        """Predict categories for multiple descriptions"""
        if not self.is_trained:
            return [], []
        
        X = self.vectorizer.transform(descriptions)
        predictions = self.model.predict(X)
        probabilities = [max(prob) for prob in self.model.predict_proba(X)]
        
        categories = self.label_encoder.inverse_transform(predictions)
        return categories.tolist(), probabilities

def main():
    st.title(" AI-Powered Expense Classifier")
    st.markdown("**Automatically categorize your expenses using machine learning**")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = ExpenseClassifier()
        st.session_state.training_data = None
    
    # Sidebar
    st.sidebar.title(" Configuration")
    
    # Model training section
    st.sidebar.subheader(" Model Training")
    
    if st.sidebar.button(" Generate Training Data & Train Model"):
        with st.spinner("Generating training data and training model..."):
            # Generate training data
            st.session_state.training_data = st.session_state.classifier.generate_sample_data(2000)
            
            # Train model
            results = st.session_state.classifier.train_model(st.session_state.training_data)
            st.session_state.training_results = results
            
            st.sidebar.success(" Model trained successfully!")
            st.sidebar.metric("Training Accuracy", f"{results['train_accuracy']:.2%}")
            st.sidebar.metric("Test Accuracy", f"{results['test_accuracy']:.2%}")
    
    # Main interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Home", " Single Prediction", " Batch Processing", 
        " Analytics", " Data Management"
    ])
    
    with tab1:
        st.markdown("### Welcome to the Expense Classifier! ")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4> Getting Started</h4>
            <ol>
            <li><strong>Train the Model:</strong> Click "Generate Training Data & Train Model" in the sidebar</li>
            <li><strong>Classify Expenses:</strong> Use the "Single Prediction" tab for individual expenses</li>
            <li><strong>Batch Processing:</strong> Upload CSV files for bulk classification</li>
            <li><strong>View Analytics:</strong> Explore your spending patterns in the Analytics tab</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("####  Supported Categories")
            for i, category in enumerate(st.session_state.classifier.categories, 1):
                st.markdown(f"{i}. {category}")
        
        # Model status
        if st.session_state.classifier.is_trained:
            st.markdown("""
            <div class="success-box">
            <h4> Model Status: Ready</h4>
            <p>Your model is trained and ready to classify expenses!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <h4> Model Status: Not Trained</h4>
            <p>Please train the model using the sidebar to start classifying expenses.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("###  Single Expense Classification")
        
        if not st.session_state.classifier.is_trained:
            st.warning(" Please train the model first using the sidebar.")
            return
        
        # Input form
        with st.form("single_prediction"):
            st.markdown("#### Enter Expense Details")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                description = st.text_input(
                    "Expense Description *",
                    placeholder="e.g., Starbucks coffee, Gas station, Grocery shopping"
                )
            
            with col2:
                amount = st.number_input(
                    "Amount ($)",
                    min_value=0.01,
                    value=50.0,
                    step=0.01
                )
            
            submitted = st.form_submit_button(" Classify Expense")
        
        if submitted and description:
            category, probability = st.session_state.classifier.predict_category(description)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(" Predicted Category", category)
            
            with col2:
                st.metric(" Confidence", f"{probability:.1%}")
            
            with col3:
                st.metric(" Amount", f"${amount:.2f}")
            
            # Confidence indicator
            if probability > 0.8:
                st.success(" High confidence prediction!")
            elif probability > 0.6:
                st.info(" Good confidence prediction")
            else:
                st.warning(" Low confidence - please verify the category")
    
    with tab3:
        st.markdown("###  Batch Processing")
        
        if not st.session_state.classifier.is_trained:
            st.warning(" Please train the model first using the sidebar.")
            return
        
        st.markdown("#### Upload CSV File")
        st.markdown("Your CSV should have columns: `date`, `description`, `amount`")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file with expense data"
        )
        
        # Sample data download
        if st.button(" Download Sample CSV Template"):
            sample_data = pd.DataFrame({
                'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
                'description': ['Starbucks coffee', 'Gas station', 'Grocery store'],
                'amount': [4.50, 35.00, 85.30]
            })
            csv = sample_data.to_csv(index=False)
            st.download_button(
                "Download Template",
                csv,
                "expense_template.csv",
                "text/csv"
            )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df = pd.read_csv(uploaded_file)
                
                st.markdown("####  Uploaded Data Preview")
                st.dataframe(df.head())
                
                if st.button(" Classify All Expenses"):
                    with st.spinner("Classifying expenses..."):
                        # Predict categories
                        categories, probabilities = st.session_state.classifier.predict_batch(
                            df['description'].tolist()
                        )
                        
                        # Add predictions to dataframe
                        df['predicted_category'] = categories
                        df['confidence'] = probabilities
                        
                        # Display results
                        st.markdown("####  Classification Results")
                        st.dataframe(df)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Expenses", len(df))
                        
                        with col2:
                            avg_confidence = np.mean(probabilities)
                            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                        
                        with col3:
                            total_amount = df['amount'].sum()
                            st.metric("Total Amount", f"${total_amount:.2f}")
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            " Download Results",
                            csv,
                            "classified_expenses.csv",
                            "text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab4:
        st.markdown("###  Expense Analytics")
        
        if st.session_state.training_data is None:
            st.warning(" Please generate training data first to view analytics.")
            return
        
        data = st.session_state.training_data.copy()
        data['month'] = pd.to_datetime(data['date']).dt.to_period('M')
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Expenses", f"{len(data):,}")
        
        with col2:
            st.metric("Total Amount", f"${data['amount'].sum():,.2f}")
        
        with col3:
            st.metric("Avg Amount", f"${data['amount'].mean():.2f}")
        
        with col4:
            st.metric("Categories", len(data['category'].unique()))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            st.markdown("####  Spending by Category")
            category_spending = data.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=category_spending.values,
                y=category_spending.index,
                orientation='h',
                title="Total Spending by Category"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly trend
            st.markdown("####  Monthly Spending Trend")
            monthly_spending = data.groupby('month')['amount'].sum()
            
            fig = px.line(
                x=monthly_spending.index.astype(str),
                y=monthly_spending.values,
                title="Monthly Spending Trend"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Category pie chart
        st.markdown("####  Category Distribution")
        fig = px.pie(
            values=category_spending.values,
            names=category_spending.index,
            title="Expense Distribution by Category"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("###  Data Management")
        
        # Export training data
        if st.session_state.training_data is not None:
            st.markdown("####  Export Training Data")
            csv = st.session_state.training_data.to_csv(index=False)
            st.download_button(
                " Download Training Data",
                csv,
                "training_data.csv",
                "text/csv"
            )
            
            st.markdown("####  Training Data Overview")
            st.dataframe(st.session_state.training_data.head())
            
            # Data statistics
            st.markdown("####  Data Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Category Counts:**")
                category_counts = st.session_state.training_data['category'].value_counts()
                for cat, count in category_counts.items():
                    st.write(f"• {cat}: {count}")
            
            with col2:
                st.markdown("**Amount Statistics:**")
                st.write(st.session_state.training_data['amount'].describe())
        
        # Model information
        st.markdown("####  Model Information")
        if st.session_state.classifier.is_trained:
            st.success(" Model is trained and ready")
            if 'training_results' in st.session_state:
                results = st.session_state.training_results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Accuracy", f"{results['train_accuracy']:.2%}")
                with col2:
                    st.metric("Test Accuracy", f"{results['test_accuracy']:.2%}")
        else:
            st.info("Model not trained yet")

if __name__ == "__main__":
    main()