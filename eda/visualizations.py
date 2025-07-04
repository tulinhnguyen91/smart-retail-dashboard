import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose # Import này cần thiết cho plot_time_series_decomposition

def plot_store_performance_ranking(df_transaction):
    """
    Vẽ biểu đồ xếp hạng hiệu suất cửa hàng theo số lượng sản phẩm bán ra.

    Args:
        df_transaction (pd.DataFrame): DataFrame chứa dữ liệu giao dịch.

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    store_performance = df_transaction.groupby('Store_Location')['Quantity_Sold'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=store_performance.values, y=store_performance.index, palette='viridis', ax=ax)
    ax.set_title('Store Performance Ranking by Units Sold', fontsize=16)
    ax.set_xlabel('Total Units Sold')
    ax.set_ylabel('Store Location')
    plt.tight_layout()
    return fig

def plot_monthly_revenue(df):
    """
    Vẽ biểu đồ doanh thu hàng tháng.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý (có cột 'Date' và 'Revenue').

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    df['Date'] = pd.to_datetime(df['Date']) # Đảm bảo cột Date là datetime
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    monthly_revenue = df.groupby('YearMonth')['Revenue'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(monthly_revenue['YearMonth'], monthly_revenue['Revenue'], marker='o', color='green', linewidth=2)
    ax.set_title('Monthly Revenue (All Brands) 2023-2024 (Million VND)', fontsize=15)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Revenue (Million VND)', fontsize=13)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    return fig

def plot_total_units_sold_by_month(df):
    """
    Vẽ biểu đồ tổng số sản phẩm bán ra theo tháng.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý (có cột 'Year', 'Month', 'Quantity_Sold').

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    monthly = df.groupby(['Year', 'Month'])['Quantity_Sold'].sum().reset_index()
    monthly['YearMonth'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=monthly, x='YearMonth', y='Quantity_Sold', marker='o', color='#0057B8', ax=ax)
    ax.set_title('Total Units Sold by Month')
    ax.set_ylabel('Units Sold')
    ax.set_xlabel('Month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_quantity_sold_by_brand_per_store(df_transaction):
    """
    Vẽ biểu đồ số lượng bán theo thương hiệu cho mỗi cửa hàng.

    Args:
        df_transaction (pd.DataFrame): DataFrame chứa dữ liệu giao dịch.

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    brand_store = df_transaction.groupby(['Store_ID', 'Brand'])['Quantity_Sold'].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    brand_store.plot(kind='bar', width=0.8, ax=ax)
    ax.set_xlabel('Store ID')
    ax.set_ylabel('Quantity Sold')
    ax.set_title('Quantity Sold by Brand for Each Store')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Brand')
    plt.tight_layout()
    return fig

def plot_apple_vs_samsung_sales_by_year(df):
    """
    Vẽ biểu đồ tổng số sản phẩm Apple vs Samsung bán ra theo năm.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý (có cột 'Year', 'Brand', 'Quantity_Sold').

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    brand_year = df.groupby(['Year', 'Brand'])['Quantity_Sold'].sum().reset_index()
    brand_year_pivot = brand_year.pivot(index='Year', columns='Brand', values='Quantity_Sold').fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    brand_year_pivot.plot(kind='bar', width=0.8, color=['#0057B8', '#B22222'], ax=ax) # Samsung - Blue, Apple - Red
    ax.set_title('Total Units Sold of Apple vs Samsung by Year', fontsize=15)
    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Total Units Sold', fontsize=13)
    plt.xticks(rotation=0)
    plt.legend(title='Brand')
    plt.tight_layout()
    return fig

def plot_monthly_units_sold_by_performance_tier(df_original):
    """
    Vẽ biểu đồ xu hướng số lượng sản phẩm bán ra hàng tháng theo cấp hiệu suất cửa hàng.

    Args:
        df_original (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý.

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    df = df_original.copy() # Làm việc trên bản sao để tránh thay đổi df_original
    store_perf = df.groupby('Store_Location')['Quantity_Sold'].sum().sort_values(ascending=False)
    tier_labels = ['High', 'Medium', 'Low']
    store_perf_tier = pd.qcut(store_perf, q=3, labels=tier_labels, duplicates='drop') # 'drop' handles non-unique bin edges

    df['Performance_Tier'] = df['Store_Location'].map(store_perf_tier)
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')

    monthly_tier_trend = df.groupby(['YearMonth', 'Performance_Tier'])['Quantity_Sold'].sum().reset_index()
    # Sắp xếp lại Performance_Tier để hiển thị đúng thứ tự trên legend/plot nếu cần
    monthly_tier_trend['Performance_Tier'] = pd.Categorical(
        monthly_tier_trend['Performance_Tier'], categories=tier_labels, ordered=True
    )
    monthly_tier_trend = monthly_tier_trend.sort_values(['YearMonth', 'Performance_Tier'])


    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=monthly_tier_trend,
        x=monthly_tier_trend['YearMonth'].astype(str),
        y='Quantity_Sold',
        hue='Performance_Tier',
        marker='o',
        palette=['#B22222', '#0057B8', '#808080'], # High-Red, Medium-Blue, Low-Gray
        ax=ax
    )
    ax.set_title('Monthly Units Sold Trend by Performance Tier')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Units Sold')
    plt.xticks(rotation=45)
    plt.legend(title='Performance Tier')
    plt.tight_layout()
    return fig

def plot_quantity_sold_by_location_tier(df_transaction_original):
    """
    Vẽ biểu đồ tổng và trung bình số lượng bán theo cấp độ vị trí cửa hàng.

    Args:
        df_transaction_original (pd.DataFrame): DataFrame chứa dữ liệu giao dịch.

    Returns:
        tuple: (matplotlib.figure.Figure, matplotlib.figure.Figure) - Hai đối tượng Figure của biểu đồ.
    """
    df_transaction = df_transaction_original.copy() # Làm việc trên bản sao

    central = ['Hoàn Kiếm', 'Đống Đa', 'Hai Bà Trưng', 'Ba Đình']
    near_central = ['Cầu Giấy', 'Thanh Xuân', 'Tây Hồ']
    suburban = ['Long Biên', 'Hoàng Mai', 'Hà Đông']

    tier_map = {loc: 'Central' for loc in central}
    tier_map.update({loc: 'Near_Central' for loc in near_central})
    tier_map.update({loc: 'Suburban' for loc in suburban})

    df_transaction['Location_Tier'] = df_transaction['Store_Location'].map(tier_map).fillna('Other')

    tier_summary = df_transaction.groupby('Location_Tier').agg(
        Total_Quantity_Sold=('Quantity_Sold', 'sum'),
        Avg_Quantity_Sold=('Quantity_Sold', 'mean')
    ).reset_index()

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(tier_summary['Location_Tier'], tier_summary['Total_Quantity_Sold'])
    ax1.set_ylabel('Total Quantity Sold')
    ax1.set_xlabel('Location Tier')
    ax1.set_title('Total Quantity Sold by Location Tier')
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(tier_summary['Location_Tier'], tier_summary['Avg_Quantity_Sold'])
    ax2.set_ylabel('Average Quantity Sold per Transaction')
    ax2.set_xlabel('Location Tier')
    ax2.set_title('Average Quantity Sold per Transaction by Location Tier')
    plt.tight_layout()
    return fig1, fig2

def plot_performance_tier_distribution_by_location_tier(df_original):
    """
    Vẽ heatmap phân phối cấp hiệu suất theo cấp độ vị trí cửa hàng.

    Args:
        df_original (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý.

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    df = df_original.copy() # Làm việc trên bản sao

    # 1. Compute each store’s total units sold
    store_perf = df.groupby('Store_Location')['Quantity_Sold'].sum().sort_values(ascending=False)

    # 2. Correctly bucket into Low/Medium/High
    tier_labels = ['Low','Medium','High']
    # Use `duplicates='drop'` to handle cases where quantiles might be identical
    store_perf_tier = pd.qcut(store_perf, q=3, labels=tier_labels, duplicates='drop')

    # 3. Map back onto your full DataFrame
    df['Performance_Tier'] = df['Store_Location'].map(store_perf_tier)

    # 4. (Re)create Location_Tier
    central = ['Hoàn Kiếm', 'Đống Đa', 'Hai Bà Trưng', 'Ba Đình']
    near_central = ['Cầu Giấy', 'Thanh Xuân', 'Tây Hồ']
    suburban = ['Long Biên', 'Hoàng Mai', 'Hà Đông']
    loc_map = {**{loc:'Central' for loc in central},
               **{loc:'Near_Central' for loc in near_central},
               **{loc:'Suburban' for loc in suburban}}
    df['Location_Tier'] = df['Store_Location'].map(loc_map)

    # Drop rows where Location_Tier or Performance_Tier couldn't be mapped
    df.dropna(subset=['Location_Tier', 'Performance_Tier'], inplace=True)

    # 5. Encode as numeric for Spearman (optional for plot, but good for correlation check)
    perf_map = {'Low':0,'Medium':1,'High':2}
    loc_map_num = {'Suburban':0,'Near_Central':1,'Central':2}
    df['Perf_Code'] = df['Performance_Tier'].map(perf_map)
    df['Loc_Code'] = df['Location_Tier'].map(loc_map_num)

    # 6. Spearman correlation (for informational print, not directly for plot)
    if not df[['Perf_Code','Loc_Code']].empty:
        # Kiểm tra để tránh lỗi nếu chỉ có 1 giá trị duy nhất sau dropna
        if df['Perf_Code'].nunique() > 1 and df['Loc_Code'].nunique() > 1:
            rho = df[['Perf_Code','Loc_Code']].corr(method='spearman').iloc[0,1]
            print(f"Spearman ρ between Performance and Location tiers: {rho:.2f}")
        else:
            print("Not enough unique values for Spearman correlation.")
    else:
        print("DataFrame is empty for Spearman correlation.")


    # 7. Crosstab and heatmap
    ct = pd.crosstab(df['Location_Tier'],
                     df['Performance_Tier'],
                     normalize='index')

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Performance Tier Distribution by Location Tier")
    ax.set_ylabel("Location Tier")
    ax.set_xlabel("Performance Tier")
    plt.tight_layout()
    return fig

def plot_promotion_activity_throughout_year(df_original):
    """
    Vẽ biểu đồ hoạt động khuyến mãi trong năm (số lượng khuyến mãi và tổng ngân sách).

    Args:
        df_original (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý.

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    df = df_original.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)

    monthly_promo = df.groupby('YearMonth').agg(
        num_promotion=('Promo_ID', lambda x: x[x != 'No Promo'].nunique()),
        total_budget=('Promo_Budget', 'sum')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.barplot(x=monthly_promo['YearMonth'], y=monthly_promo['num_promotion'], color='#0057B8', ax=ax1)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Promotions', color='#0057B8')
    ax1.tick_params(axis='y', labelcolor='#0057B8')
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    sns.lineplot(x=monthly_promo['YearMonth'], y=monthly_promo['total_budget'],
                 color='#FF7300', marker='o', linewidth=2.5, ax=ax2)
    ax2.set_ylabel('Total Promotion Budget', color='#FF7300')
    ax2.tick_params(axis='y', labelcolor='#FF7300')

    plt.title('Promotion Activity Throughout the Year')
    plt.tight_layout()
    return fig

def plot_promotion_type_frequency_by_brand(df_original):
    """
    Vẽ biểu đồ tần suất loại khuyến mãi theo thương hiệu.

    Args:
        df_original (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý.

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    df = df_original.copy()
    promo_map = {1: 'Discount', 2: 'Trade-in'}
    df['Promo_Type'] = df['Promo_Type_Code'].map(promo_map)

    promo_df = df[df['Promo_Type_Code'].isin([1,2])].copy()
    if promo_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No promotional data to display.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('Promotion Type Frequency by Brand')
        return fig

    promo_brand_counts = promo_df.groupby(['Brand', 'Promo_Type']).size().reset_index(name='Count')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=promo_brand_counts, x='Promo_Type', y='Count', hue='Brand',
                palette=['#B22222', '#0057B8'], ax=ax) # Apple - đỏ, Samsung - xanh
    ax.set_title('Promotion Type Frequency by Brand')
    ax.set_xlabel('Promotion Type')
    ax.set_ylabel('Frequency')
    ax.legend(title='Brand')
    plt.tight_layout()
    return fig

def plot_monthly_promo_budget_vs_units_sold(df_original):
    """
    Vẽ biểu đồ tương quan giữa ngân sách khuyến mãi hàng tháng và số lượng sản phẩm bán ra.

    Args:
        df_original (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý.

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    df = df_original.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')

    monthly_all = df.groupby('YearMonth')['Quantity_Sold'].sum().rename('monthly_units_sold')
    monthly_budget = df.groupby('YearMonth')['Promo_Budget'].sum().rename('monthly_promo_budget')

    monthly = pd.concat([monthly_all, monthly_budget], axis=1).fillna(0).reset_index()

    # Pearson’s r
    if len(monthly) > 1:
        # Kiểm tra để tránh lỗi nếu chỉ có 1 giá trị duy nhất sau fillna
        if monthly['monthly_promo_budget'].nunique() > 1 and monthly['monthly_units_sold'].nunique() > 1:
            r, _ = pearsonr(monthly['monthly_promo_budget'], monthly['monthly_units_sold'])
            print(f"Pearson’s r = {r:.2f}")
        else:
            print("Not enough unique values for Pearson correlation.")
    else:
        print("Not enough data points for Pearson correlation.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=monthly, x='monthly_promo_budget', y='monthly_units_sold',
                scatter_kws={'s': 70, 'color': '#0057B8'},
                line_kws={'color': '#B22222', 'lw': 2}, ax=ax)
    ax.set_title('Correlation: Monthly Promo Budget vs Units Sold')
    ax.set_xlabel('Monthly Promo Budget (VND)')
    ax.set_ylabel('Monthly Units Sold')
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df_original):
    """
    Vẽ heatmap ma trận tương quan của các biến số.

    Args:
        df_original (pd.DataFrame): DataFrame chứa dữ liệu đã được xử lý.

    Returns:
        matplotlib.figure.Figure: Đối tượng Figure của biểu đồ.
    """
    df = df_original.copy()
    df['Revenue'] = df['Quantity_Sold'] * df['Price'] # Recompute for consistency

    numerical_columns = [
        'Quantity_Sold', 'Price', 'Stock_Level', 'Reorder_Threshold',
        'Revenue', 'Year', 'Month', 'Quarter', 'Promo_Type_Code',
        'Promo_Budget', 'Store_Size'
    ]
    df_numeric = df[numerical_columns].copy()

    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric.dropna(inplace=True)

    if df_numeric.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Không có dữ liệu số để tạo heatmap.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('Correlation Heatmap of Numerical Variables')
        return fig
    
    corr_matrix = df_numeric.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1, center=0,
                square=True, fmt='.2f', annot_kws={'size': 10},
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    ax.set_title('Correlation Heatmap of Numerical Variables', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    return fig

# --- Các hàm EDA mới được thêm vào ---

def plot_sales_distribution(df):
    """
    Vẽ biểu đồ phân phối của Quantity_Sold.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Quantity_Sold'], kde=True, bins=30, ax=ax)
    ax.set_title('Phân phối số lượng bán (Quantity_Sold)')
    ax.set_xlabel('Quantity Sold')
    ax.set_ylabel('Tần suất')
    plt.tight_layout()
    return fig

def get_top_n_products(df, n=10):
    """
    Lấy danh sách N sản phẩm bán chạy nhất theo tổng số lượng.
    """
    top_products = df.groupby('Product_Name')['Quantity_Sold'].sum().nlargest(n).reset_index()
    return top_products

def get_descriptive_stats(df_original, columns_to_describe=None):
    """
    Trả về bảng thống kê mô tả cho các cột được chọn.
    Args:
        df_original (pd.DataFrame): DataFrame đầu vào.
        columns_to_describe (list, optional): Danh sách các cột muốn thống kê.
                                                Nếu None, sẽ thống kê tất cả các cột số.
    Returns:
        pd.DataFrame: DataFrame chứa thống kê mô tả.
    """
    df = df_original.copy()
    if columns_to_describe:
        df_stats = df[columns_to_describe].describe()
    else:
        df_stats = df.describe()
    return df_stats

def get_top_n_products_by_revenue(df_original, n=10):
    """
    Trả về bảng Top N sản phẩm bán chạy nhất theo Doanh thu.
    Args:
        df_original (pd.DataFrame): DataFrame đầu vào (có cột 'Product_Name' và 'Revenue').
        n (int): Số lượng sản phẩm top muốn hiển thị.
    Returns:
        pd.DataFrame: DataFrame chứa Top N sản phẩm.
    """
    df = df_original.copy()
    if 'Revenue' not in df.columns:
        df['Revenue'] = df['Quantity_Sold'] * df['Price'] # Đảm bảo cột Revenue tồn tại
    top_products_revenue = df.groupby('Product_Name')['Revenue'].sum().nlargest(n).reset_index()
    top_products_revenue.columns = ['Product_Name', 'Total_Revenue']
    return top_products_revenue

def plot_time_series_decomposition(df_original, product_name):
    df_product = df_original[df_original['Product_Name'] == product_name].copy()
    df_daily = df_product.groupby('Date')['Quantity_Sold'].sum().reset_index()
    df_daily = df_daily.set_index('Date').sort_index()

    # Kiểm tra số lượng dữ liệu đủ để phân rã
    # statsmodels.tsa.seasonal.seasonal_decompose yêu cầu ít nhất 2 chu kỳ.
    # Nếu period=365, cần ít nhất 2*365 = 730 ngày dữ liệu.
    if len(df_daily) < 2 * 365:
        print(f"Không đủ dữ liệu ({len(df_daily)} ngày) cho phân rã chuỗi thời gian của {product_name}. Cần ít nhất 730 ngày.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Không đủ dữ liệu cho phân rã chuỗi thời gian.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'Phân rã chuỗi thời gian cho {product_name}')
        return fig

    try:
        # Adjust model based on data frequency, 'multiplicative' often good for sales
        result = seasonal_decompose(df_daily['Quantity_Sold'], model='multiplicative', period=365)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        result.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        result.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        result.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')
        result.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')
        fig.suptitle(f'Phân rã chuỗi thời gian cho {product_name}', y=1.02)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Lỗi khi phân rã chuỗi thời gian cho {product_name}: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Lỗi: {e}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'Phân rã chuỗi thời gian cho {product_name} (Lỗi)')
        return fig

