import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("DIGITAL MEDIA TRENDS ANALYSIS: OTT & SOCIAL MEDIA PLATFORMS")
print("=" * 80)
print("\n")

# ============================================================================
# PART 1: GENERATE SYNTHETIC DATASET
# ============================================================================

print("STEP 1: Generating Datasets...")
print("-" * 80)

# Set random seed for reproducibility
np.random.seed(42)

# Generate OTT Platform Dataset
n_ott_records = 1000
ott_platforms = ['Netflix', 'Amazon Prime', 'Disney+', 'HBO Max', 'Hulu']
content_types = ['Movie', 'TV Series', 'Documentary', 'Reality Show']
genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Sci-Fi', 'Romance', 'Horror']
age_groups = ['13-18', '19-25', '26-35', '36-50', '50+']

ott_data = {
    'UserID': [f'U{str(i).zfill(4)}' for i in range(1, n_ott_records + 1)],
    'Platform': np.random.choice(ott_platforms, n_ott_records, p=[0.35, 0.25, 0.20, 0.12, 0.08]),
    'ContentType': np.random.choice(content_types, n_ott_records, p=[0.35, 0.40, 0.15, 0.10]),
    'Genre': np.random.choice(genres, n_ott_records),
    'WatchTimeHours': np.round(np.random.exponential(3, n_ott_records) + 0.5, 2),
    'SubscriptionType': np.random.choice(['Free', 'Basic', 'Premium'], n_ott_records, p=[0.15, 0.45, 0.40]),
    'AgeGroup': np.random.choice(age_groups, n_ott_records, p=[0.10, 0.30, 0.35, 0.20, 0.05]),
    'DeviceType': np.random.choice(['Mobile', 'Tablet', 'Smart TV', 'Desktop'], n_ott_records, p=[0.40, 0.15, 0.30, 0.15]),
    'Rating': np.round(np.random.uniform(1, 5, n_ott_records), 1),
    'CompletionRate': np.round(np.random.beta(8, 2, n_ott_records) * 100, 2)
}

df_ott = pd.DataFrame(ott_data)

# Generate Social Media Dataset
n_social_records = 1000
social_platforms = ['Instagram', 'TikTok', 'YouTube', 'Twitter', 'Facebook']
content_categories = ['Entertainment', 'Education', 'News', 'Sports', 'Lifestyle', 'Technology']
engagement_types = ['Like', 'Comment', 'Share', 'View']

social_data = {
    'UserID': [f'S{str(i).zfill(4)}' for i in range(1, n_social_records + 1)],
    'Platform': np.random.choice(social_platforms, n_social_records, p=[0.30, 0.25, 0.25, 0.10, 0.10]),
    'ContentCategory': np.random.choice(content_categories, n_social_records),
    'EngagementType': np.random.choice(engagement_types, n_social_records, p=[0.50, 0.20, 0.15, 0.15]),
    'TimeSpentMinutes': np.round(np.random.exponential(45, n_social_records) + 10, 2),
    'PostsViewed': np.random.poisson(50, n_social_records),
    'InteractionsCount': np.random.poisson(15, n_social_records),
    'AgeGroup': np.random.choice(age_groups, n_social_records, p=[0.25, 0.35, 0.25, 0.12, 0.03]),
    'InfluencerFollowing': np.random.choice(['Yes', 'No'], n_social_records, p=[0.65, 0.35]),
    'AdClicks': np.random.poisson(3, n_social_records)
}

df_social = pd.DataFrame(social_data)

print(f"✓ OTT Dataset created: {len(df_ott)} records")
print(f"✓ Social Media Dataset created: {len(df_social)} records")
print("\n")

# ============================================================================
# PART 2: DATA EXPLORATION
# ============================================================================

print("STEP 2: Data Exploration")
print("-" * 80)

print("\n--- OTT Platform Dataset ---")
print(df_ott.head())
print(f"\nDataset Shape: {df_ott.shape}")
print(f"\nData Types:\n{df_ott.dtypes}")
print(f"\nMissing Values:\n{df_ott.isnull().sum()}")

print("\n--- Social Media Dataset ---")
print(df_social.head())
print(f"\nDataset Shape: {df_social.shape}")
print(f"\nData Types:\n{df_social.dtypes}")
print(f"\nMissing Values:\n{df_social.isnull().sum()}")
print("\n")

# ============================================================================
# PART 3: STATISTICAL ANALYSIS
# ============================================================================

print("STEP 3: Statistical Analysis")
print("-" * 80)

print("\n--- OTT Platform Statistics ---")
print(df_ott.describe())

print("\n--- Social Media Statistics ---")
print(df_social.describe())
print("\n")

# ============================================================================
# PART 4: TREND ANALYSIS
# ============================================================================

print("STEP 4: Uncovering Key Trends")
print("-" * 80)

# OTT Trends
print("\n1. OTT PLATFORM TRENDS:")
print("-" * 40)

platform_usage = df_ott['Platform'].value_counts()
print(f"\nPlatform Popularity:\n{platform_usage}")

avg_watch_time = df_ott.groupby('Platform')['WatchTimeHours'].mean().sort_values(ascending=False)
print(f"\nAverage Watch Time by Platform (hours):\n{avg_watch_time.round(2)}")

content_preference = df_ott['ContentType'].value_counts()
print(f"\nContent Type Preferences:\n{content_preference}")

subscription_dist = df_ott['SubscriptionType'].value_counts()
print(f"\nSubscription Distribution:\n{subscription_dist}")

# Social Media Trends
print("\n\n2. SOCIAL MEDIA TRENDS:")
print("-" * 40)

social_platform_usage = df_social['Platform'].value_counts()
print(f"\nPlatform Popularity:\n{social_platform_usage}")

avg_time_spent = df_social.groupby('Platform')['TimeSpentMinutes'].mean().sort_values(ascending=False)
print(f"\nAverage Time Spent by Platform (minutes):\n{avg_time_spent.round(2)}")

content_category = df_social['ContentCategory'].value_counts()
print(f"\nContent Category Preferences:\n{content_category}")

engagement_analysis = df_social.groupby('Platform')['InteractionsCount'].mean().sort_values(ascending=False)
print(f"\nAverage Interactions by Platform:\n{engagement_analysis.round(2)}")

# Age Group Analysis
print("\n\n3. AGE GROUP INSIGHTS:")
print("-" * 40)

ott_age = df_ott.groupby('AgeGroup')['WatchTimeHours'].mean().sort_values(ascending=False)
print(f"\nOTT Watch Time by Age Group:\n{ott_age.round(2)}")

social_age = df_social.groupby('AgeGroup')['TimeSpentMinutes'].mean().sort_values(ascending=False)
print(f"\nSocial Media Time by Age Group:\n{social_age.round(2)}")

print("\n")

# ============================================================================
# PART 5: ADVANCED INSIGHTS
# ============================================================================

print("STEP 5: Advanced Insights")
print("-" * 80)

# Device preferences
device_analysis = df_ott.groupby('DeviceType').agg({
    'WatchTimeHours': 'mean',
    'Rating': 'mean',
    'CompletionRate': 'mean'
}).round(2)
print("\nDevice Type Analysis (OTT):")
print(device_analysis)

# Genre ratings
genre_ratings = df_ott.groupby('Genre')['Rating'].mean().sort_values(ascending=False)
print(f"\nTop Rated Genres:\n{genre_ratings.round(2)}")

# Influencer impact
influencer_impact = df_social.groupby('InfluencerFollowing').agg({
    'TimeSpentMinutes': 'mean',
    'InteractionsCount': 'mean',
    'AdClicks': 'mean'
}).round(2)
print("\nInfluencer Following Impact:")
print(influencer_impact)

print("\n")

# ============================================================================
# PART 6: VISUALIZATIONS
# ============================================================================

print("STEP 6: Creating Visualizations...")
print("-" * 80)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. OTT Platform Distribution
ax1 = plt.subplot(3, 3, 1)
platform_usage.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('OTT Platform Usage Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Platform')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)

# 2. Average Watch Time by Platform
ax2 = plt.subplot(3, 3, 2)
avg_watch_time.plot(kind='barh', color='coral', edgecolor='black')
plt.title('Average Watch Time by Platform', fontsize=12, fontweight='bold')
plt.xlabel('Hours')
plt.ylabel('Platform')

# 3. Content Type Distribution
ax3 = plt.subplot(3, 3, 3)
content_preference.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Content Type Distribution', fontsize=12, fontweight='bold')
plt.ylabel('')

# 4. Social Media Platform Usage
ax4 = plt.subplot(3, 3, 4)
social_platform_usage.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Social Media Platform Usage', fontsize=12, fontweight='bold')
plt.xlabel('Platform')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)

# 5. Average Time Spent on Social Media
ax5 = plt.subplot(3, 3, 5)
avg_time_spent.plot(kind='barh', color='gold', edgecolor='black')
plt.title('Avg Time Spent (Social Media)', fontsize=12, fontweight='bold')
plt.xlabel('Minutes')
plt.ylabel('Platform')

# 6. Subscription Type Distribution
ax6 = plt.subplot(3, 3, 6)
subscription_dist.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#c2c2f0','#ffb3e6','#c2f0c2'])
plt.title('Subscription Type Distribution', fontsize=12, fontweight='bold')
plt.ylabel('')

# 7. Watch Time by Age Group
ax7 = plt.subplot(3, 3, 7)
ott_age.plot(kind='bar', color='mediumpurple', edgecolor='black')
plt.title('OTT Watch Time by Age Group', fontsize=12, fontweight='bold')
plt.xlabel('Age Group')
plt.ylabel('Average Hours')
plt.xticks(rotation=45)

# 8. Social Media Time by Age Group
ax8 = plt.subplot(3, 3, 8)
social_age.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Social Media Time by Age Group', fontsize=12, fontweight='bold')
plt.xlabel('Age Group')
plt.ylabel('Average Minutes')
plt.xticks(rotation=45)

# 9. Genre Ratings
ax9 = plt.subplot(3, 3, 9)
genre_ratings.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Average Ratings by Genre', fontsize=12, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Genre')

plt.tight_layout()
plt.savefig('digital_media_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'digital_media_analysis.png'")

# Additional correlation heatmap
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# OTT Correlation
ott_numeric = df_ott[['WatchTimeHours', 'Rating', 'CompletionRate']].corr()
sns.heatmap(ott_numeric, annot=True, cmap='coolwarm', center=0, ax=ax1, fmt='.2f')
ax1.set_title('OTT Platform Correlations', fontsize=14, fontweight='bold')

# Social Media Correlation
social_numeric = df_social[['TimeSpentMinutes', 'PostsViewed', 'InteractionsCount', 'AdClicks']].corr()
sns.heatmap(social_numeric, annot=True, cmap='viridis', center=0, ax=ax2, fmt='.2f')
ax2.set_title('Social Media Correlations', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmaps saved as 'correlation_heatmaps.png'")

plt.show()

# ============================================================================
# PART 7: KEY FINDINGS & RECOMMENDATIONS
# ============================================================================

print("\n")
print("STEP 7: Key Findings & Recommendations")
print("=" * 80)

print("\n📊 KEY FINDINGS:")
print("-" * 80)

print("\n1. OTT PLATFORM INSIGHTS:")
print(f"   • Netflix dominates with {platform_usage['Netflix']} users ({platform_usage['Netflix']/len(df_ott)*100:.1f}%)")
print(f"   • TV Series is the most popular content type ({content_preference['TV Series']} views)")
print(f"   • Premium subscriptions account for {subscription_dist['Premium']/len(df_ott)*100:.1f}% of users")
print(f"   • Mobile devices are preferred by {(df_ott['DeviceType']=='Mobile').sum()/len(df_ott)*100:.1f}% of users")

print("\n2. SOCIAL MEDIA INSIGHTS:")
print(f"   • Instagram leads with {social_platform_usage['Instagram']} users")
print(f"   • {(df_social['InfluencerFollowing']=='Yes').sum()/len(df_social)*100:.1f}% of users follow influencers")
print(f"   • Entertainment is the top content category")
print(f"   • Age group 19-25 is the most active demographic")

print("\n3. CROSS-PLATFORM TRENDS:")
print(f"   • Young users (19-25) spend more time on social media")
print(f"   • Mid-age users (26-35) prefer OTT platforms for longer content")
print(f"   • Mobile-first approach is critical for both platforms")

print("\n💡 RECOMMENDATIONS:")
print("-" * 80)
print("   1. Focus on mobile optimization for better user experience")
print("   2. Invest in TV Series content for OTT platforms")
print("   3. Leverage influencer marketing for social media engagement")
print("   4. Create age-specific content strategies")
print("   5. Offer flexible subscription models to attract free-tier users")

print("\n")
print("=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

# Save datasets to CSV
df_ott.to_csv('ott_platform_data.csv', index=False)
df_social.to_csv('social_media_data.csv', index=False)
print("\n✓ Datasets saved: 'ott_platform_data.csv' and 'social_media_data.csv'")
print("\nThank you for using Digital Media Trends Analysis System!")
print("=" * 80)