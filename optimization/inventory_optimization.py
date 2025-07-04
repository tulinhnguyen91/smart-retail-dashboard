import pandas as pd
from pulp import *

def run_inventory_optimization(df_final_processed, df_store_info_original, # Đổi tên tham số để rõ ràng hơn
                               warehouse_supply_limit=1000,
                               storage_capacity_per_sqm=0.5):
    """
    Thực hiện tối ưu hóa tồn kho từ kho tổng đến các cửa hàng.

    Args:
        df_final_processed (pd.DataFrame): DataFrame chứa dữ liệu giao dịch ĐÃ XỬ LÝ.
        df_store_info_original (pd.DataFrame): DataFrame chứa thông tin cửa hàng GỐC.
        warehouse_supply_limit (int): Tổng số sản phẩm tối đa có thể phân phối từ kho tổng.
        storage_capacity_per_sqm (float): Sức chứa sản phẩm trên mỗi mét vuông của cửa hàng.

    Returns:
        pd.DataFrame: DataFrame chứa kết quả tối ưu hóa tồn kho (số lượng điều chỉnh).
                      Trả về DataFrame rỗng nếu không thể thực hiện tối ưu.
    """
    print("\n--- Running Inventory Optimization ---")

    # --- 1. Load and Prepare Data ---
    # KHÔNG CẦN ĐỌC LẠI TỪ FILE. SỬ DỤNG df_final_processed VÀ df_store_info_original ĐÃ ĐƯỢC TRUYỀN VÀO.
    df_final = df_final_processed.copy()
    df_store_info = df_store_info_original.copy()

    df_final['Date'] = pd.to_datetime(df_final['Date'])
    latest_date = df_final['Date'].max()
    df_snapshot = df_final[df_final['Date'] == latest_date].copy()

    df_inventory_data = df_snapshot.groupby(['Store_ID', 'Product_Name']).agg(
        Current_Stock=('Stock_Level', 'sum'),
        Reorder_Threshold=('Reorder_Threshold', 'mean')
    ).reset_index()

    df_store_info_relevant = df_store_info[['Store_ID', 'Store_Size']].drop_duplicates(subset=['Store_ID'])
    df_inventory_data = pd.merge(df_inventory_data, df_store_info_relevant, on='Store_ID', how='left')

    df_inventory_data.fillna({
        'Current_Stock': 0,
        'Reorder_Threshold': 0,
        'Store_Size': 500
    }, inplace=True)

    if df_inventory_data.empty:
        print("  Warning: Inventory data is empty. Skipping optimization.")
        return pd.DataFrame()

    # Thêm cột Initial_Shortage để dễ kiểm tra và cho mục đích tối ưu
    df_inventory_data['Initial_Shortage'] = df_inventory_data.apply(
        lambda row: max(0, row['Reorder_Threshold'] - row['Current_Stock']), axis=1
    )

    # Các dòng print để debug (có thể xóa sau khi chạy thành công)
    print(f"Tổng số bản ghi tồn kho trong snapshot: {len(df_inventory_data)}")
    print(f"Tổng số nhu cầu bổ sung ban đầu (tổng Initial_Shortage): {df_inventory_data['Initial_Shortage'].sum()}")
    print("5 dòng đầu tiên của dữ liệu tồn kho (df_inventory_data):")
    print(df_inventory_data.head())


    stores = df_inventory_data['Store_ID'].unique().tolist()
    products = df_inventory_data['Product_Name'].unique().tolist()
    inventory_data_dict = {(r['Store_ID'], r['Product_Name']): r.to_dict() for _, r in df_inventory_data.iterrows()}

    # --- 2. PuLP Model Setup & Solve ---
    prob = LpProblem("Warehouse_Inventory_Allocation", LpMaximize)

    adjustment_vars = LpVariable.dicts("Adjustment",
                                       [(s, p) for s in stores for p in products if (s, p) in inventory_data_dict],
                                       lowBound=0, cat='Integer')

    priority_score = {}
    for (s, p), data in inventory_data_dict.items():
        score = 0.001
        if data['Current_Stock'] < data['Reorder_Threshold']:
            score = max(score, (data['Reorder_Threshold'] - data['Current_Stock']))
        priority_score[(s, p)] = score

    prob += lpSum(adjustment_vars[(s, p)] * priority_score[(s, p)]
                  for (s, p) in adjustment_vars), "Maximize_Weighted_Allocated_Units"

    for s in stores:
        # Đảm bảo chỉ lấy các sản phẩm có trong inventory_data_dict cho cửa hàng s
        products_in_store_s = [p for p in products if (s, p) in inventory_data_dict]
        if not products_in_store_s: # Bỏ qua nếu cửa hàng không có sản phẩm nào trong snapshot
            continue

        total_adjusted_stock_at_store = lpSum(inventory_data_dict[(s, p)]['Current_Stock'] + adjustment_vars[(s, p)]
                                              for p in products_in_store_s)
        # Lấy store_size từ một sản phẩm bất kỳ trong cửa hàng đó
        store_size = inventory_data_dict[(s, products_in_store_s[0])]['Store_Size']
        prob += total_adjusted_stock_at_store <= store_size * storage_capacity_per_sqm, f"Storage_Limit_Store_{s}"

    prob += lpSum(adjustment_vars[(s, p)] for (s, p) in adjustment_vars) <= warehouse_supply_limit, "Total_Warehouse_Supply_Limit"

    try:
        prob.solve(PULP_CBC_CMD(msg=0))
        print(f"  Solver Status: {LpStatus[prob.status]}")
    except Exception as e:
        print(f"  Error solving PuLP problem: {e}")
        return pd.DataFrame()

    # --- 3. Display Results ---
    inventory_optimization_results = []
    total_adjusted_units_actual = 0

    # Chỉ lặp qua các cặp (s,p) thực sự có trong adjustment_vars
    for (s, p) in adjustment_vars:
        adjustment = adjustment_vars[(s, p)].varValue
        # Chỉ thêm vào kết quả nếu có điều chỉnh hoặc nếu bạn muốn xem tất cả các dòng
        if adjustment is not None and adjustment > 0: # Chỉ hiển thị các điều chỉnh > 0
            current_stock = inventory_data_dict[(s, p)]['Current_Stock']
            reorder_threshold = inventory_data_dict[(s, p)]['Reorder_Threshold']
            store_size = inventory_data_dict[(s, p)]['Store_Size']

            final_stock = current_stock + adjustment
            total_adjusted_units_actual += adjustment

            inventory_optimization_results.append({
                'Store_ID': s,
                'Product_Name': p,
                'Current_Stock': current_stock,
                'Reorder_Threshold': reorder_threshold,
                'Store_Size': store_size,
                'Adjustment_Units_From_Warehouse': adjustment,
                'Final_Stock_After_Adjustment': final_stock,
                'Met_Threshold_After_Opt': 'Yes' if final_stock >= reorder_threshold else 'No',
                'Remaining_Shortage_After_Opt': max(0, reorder_threshold - final_stock)
            })

    df_inventory_results = pd.DataFrame(inventory_optimization_results)
    if not df_inventory_results.empty:
        df_inventory_results['Adjustment_Units_From_Warehouse'] = df_inventory_results['Adjustment_Units_From_Warehouse'].astype(int)
        df_inventory_results['Final_Stock_After_Adjustment'] = df_inventory_results['Final_Stock_After_Adjustment'].astype(int)
        df_inventory_results['Remaining_Shortage_After_Opt'] = df_inventory_results['Remaining_Shortage_After_Opt'].astype(int)


    print(f"  Total Units Distributed from Warehouse: {total_adjusted_units_actual:.0f} (Max allowed: {warehouse_supply_limit})")

    return df_inventory_results

