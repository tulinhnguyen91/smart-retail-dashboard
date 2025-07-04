import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import seaborn as sns

def run_supply_chain_optimization(df_final_processed, df_store_info_original, # Đổi tên tham số để rõ ràng hơn
                                  storage_capacity_per_sqm=0.5,
                                  unmet_demand_penalty=1000):
    """
    Thực hiện tối ưu hóa chuỗi cung ứng (chuyển kho giữa các cửa hàng).

    Args:
        df_final_processed (pd.DataFrame): DataFrame chứa dữ liệu giao dịch ĐÃ XỬ LÝ.
        df_store_info_original (pd.DataFrame): DataFrame chứa thông tin cửa hàng GỐC.
        storage_capacity_per_sqm (float): Sức chứa sản phẩm trên mỗi mét vuông của cửa hàng.
        unmet_demand_penalty (int): Chi phí phạt cho mỗi đơn vị nhu cầu không được đáp ứng.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame, matplotlib.figure.Figure)
               DataFrame kết quả chuyển kho, DataFrame tóm tắt nhu cầu không được đáp ứng,
               và Figure của biểu đồ chuyển kho. Trả về (None, None, None) nếu không thể tối ưu.
    """
    print("\n--- Running Supply Chain Optimization (Store-to-Store Transfers) ---")

    # --- 1. Load and Prepare Data ---
    # KHÔNG CẦN ĐỌC LẠI TỪ FILE. SỬ DỤNG df_final_processed VÀ df_store_info_original ĐÃ ĐƯỢC TRUYỀN VÀO.
    df_final = df_final_processed.copy()
    df_store_info = df_store_info_original.copy()


    df_final['Date'] = pd.to_datetime(df_final['Date'])
    latest_date = df_final['Date'].max()
    df_snapshot = df_final[df_final['Date'] == latest_date].copy()

    df_current_inventory = df_snapshot.groupby(['Store_ID', 'Product_Name']).agg(
        Current_Stock=('Stock_Level', 'sum'),
        Reorder_Threshold=('Reorder_Threshold', 'mean')
    ).reset_index()

    df_store_info_relevant = df_store_info[['Store_ID', 'Store_Size']].drop_duplicates(subset=['Store_ID'])
    df_current_inventory = pd.merge(df_current_inventory, df_store_info_relevant, on='Store_ID', how='left')

    df_current_inventory.fillna({
        'Current_Stock': 0,
        'Reorder_Threshold': 0,
        'Store_Size': 500
    }, inplace=True)

    if df_current_inventory.empty:
        print("  Warning: Current inventory data is empty. Skipping optimization.")
        return None, None, None

    df_current_inventory['Initial_Shortage'] = df_current_inventory.apply(
        lambda row: max(0, row['Reorder_Threshold'] - row['Current_Stock']), axis=1
    )
    df_current_inventory['Initial_Surplus'] = df_current_inventory.apply(
        lambda row: max(0, row['Current_Stock'] - row['Reorder_Threshold']), axis=1
    )

    # Các dòng print để debug (có thể xóa sau khi chạy thành công)
    print(f"Tổng số bản ghi tồn kho hiện tại: {len(df_current_inventory)}")
    print(f"Tổng số nhu cầu bổ sung ban đầu (tổng Initial_Shortage): {df_current_inventory['Initial_Shortage'].sum()}")
    print("5 dòng đầu tiên của dữ liệu tồn kho hiện tại (df_current_inventory):")
    print(df_current_inventory.head())


    stores = df_current_inventory['Store_ID'].unique().tolist()
    products = df_current_inventory['Product_Name'].unique().tolist()

    inventory_data_dict = {(r['Store_ID'], r['Product_Name']): r.to_dict() for _, r in df_current_inventory.iterrows()}
    store_sizes = {s_id: df_store_info_relevant[df_store_info_relevant['Store_ID'] == s_id]['Store_Size'].iloc[0]
                   for s_id in stores}

    # Mocking up Distance and Transport Cost Matrix
    np.random.seed(42)
    store_distance_mapping = {store: i for i, store in enumerate(stores)}
    num_stores = len(stores)
    distance_matrix = np.zeros((num_stores, num_stores))

    for i in range(num_stores):
        for j in range(i + 1, num_stores):
            dist = np.random.randint(10, 500)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    transport_cost_per_unit = {}
    for s_from in stores:
        for s_to in stores:
            if s_from == s_to:
                transport_cost_per_unit[(s_from, s_to)] = 0
            else:
                dist = distance_matrix[store_distance_mapping[s_from], store_distance_mapping[s_to]]
                transport_cost_per_unit[(s_from, s_to)] = dist * 0.1

    # --- 2. PuLP Model Setup & Solve ---
    prob = LpProblem("Store_to_Store_Transfers", LpMinimize)

    transfer_vars = LpVariable.dicts("Transfer",
                                     [(s_from, s_to, p)
                                      for s_from in stores for s_to in stores if s_from != s_to
                                      for p in products if (s_from, p) in inventory_data_dict and (s_to, p) in inventory_data_dict],
                                     lowBound=0, cat='Integer')

    prob += lpSum(transfer_vars[(s_from, s_to, p)] * transport_cost_per_unit[(s_from, s_to)]
                  for (s_from, s_to, p) in transfer_vars), "Minimize_Total_Transport_Cost"

    unmet_demand_vars = LpVariable.dicts("Unmet_Demand",
                                         [(s_to, p) for s_to in stores for p in products if (s_to, p) in inventory_data_dict and inventory_data_dict[(s_to, p)]['Initial_Shortage'] > 0],
                                         lowBound=0)

    prob += lpSum(unmet_demand_vars[(s_to, p)] * unmet_demand_penalty
                  for (s_to, p) in unmet_demand_vars), "Minimize_Unmet_Demand"

    for s_from in stores:
        for p in products:
            if (s_from, p) in inventory_data_dict:
                total_transferred_out = lpSum(transfer_vars[(s_from, s_to, p)]
                                              for s_to in stores if s_from != s_to and (s_from, s_to, p) in transfer_vars)
                available_for_transfer = inventory_data_dict[(s_from, p)]['Initial_Surplus']
                prob += total_transferred_out <= available_for_transfer, f"Max_Transfer_Out_Store_{s_from}_Product_{p}"

    for s in stores:
        for p in products:
            if (s, p) in inventory_data_dict:
                current_stock = inventory_data_dict[(s, p)]['Current_Stock']
                reorder_threshold = inventory_data_dict[(s, p)]['Reorder_Threshold']
                initial_shortage = inventory_data_dict[(s, p)]['Initial_Shortage']

                incoming_transfers = lpSum(transfer_vars[(s_from, s, p)]
                                           for s_from in stores if s_from != s and (s_from, s, p) in transfer_vars)
                outgoing_transfers = lpSum(transfer_vars[(s, s_to, p)]
                                           for s_to in stores if s_to != s and (s, s_to, p) in transfer_vars)

                final_stock_at_store_product = current_stock + incoming_transfers - outgoing_transfers

                if initial_shortage > 0:
                    prob += final_stock_at_store_product >= reorder_threshold - unmet_demand_vars[(s, p)], f"Meet_Demand_Store_{s}_Product_{p}"
                else:
                    prob += final_stock_at_store_product >= reorder_threshold, f"Maintain_Threshold_Store_{s}_Product_{p}"

    for s_to in stores:
        # Đảm bảo chỉ lấy các sản phẩm có trong inventory_data_dict cho cửa hàng s_to
        products_in_store_s_to = [p for p in products if (s_to, p) in inventory_data_dict]
        if not products_in_store_s_to:
            continue

        total_final_stock_at_receiving_store = lpSum(
            inventory_data_dict[(s_to, p)]['Current_Stock'] +
            lpSum(transfer_vars[(s_from, s_to, p)] for s_from in stores if s_from != s_to and (s_from, s_to, p) in transfer_vars) -
            lpSum(transfer_vars[(s_to, s_from_dummy, p)] for s_from_dummy in stores if s_to != s_from_dummy and (s_to, s_from_dummy, p) in transfer_vars)
            for p in products_in_store_s_to
        )
        store_max_capacity = store_sizes[s_to] * storage_capacity_per_sqm
        prob += total_final_stock_at_receiving_store <= store_max_capacity, f"Receiving_Capacity_Store_{s_to}"

    try:
        prob.solve(PULP_CBC_CMD(msg=0))
        print(f"  Solver Status: {LpStatus[prob.status]}")
    except Exception as e:
        print(f"  Error solving PuLP problem: {e}")
        return None, None, None

    # --- 3. Display Results ---
    transfer_results = []
    total_transferred_units = 0
    total_transport_cost = 0

    for (s_from, s_to, p) in transfer_vars:
        if transfer_vars[(s_from, s_to, p)].varValue is not None and transfer_vars[(s_from, s_to, p)].varValue > 0:
            units_transferred = transfer_vars[(s_from, s_to, p)].varValue
            cost = transport_cost_per_unit[(s_from, s_to)] * units_transferred
            total_transferred_units += units_transferred
            total_transport_cost += cost
            transfer_results.append({
                'From_Store': s_from,
                'To_Store': s_to,
                'Product_Name': p,
                'Transferred_Units': units_transferred,
                'Transport_Cost_Per_Unit': transport_cost_per_unit[(s_from, s_to)],
                'Total_Transfer_Cost': cost
            })

    df_transfer_results = pd.DataFrame(transfer_results)
    if not df_transfer_results.empty:
        df_transfer_results['Transferred_Units'] = df_transfer_results['Transferred_Units'].astype(int)
        df_transfer_results['Total_Transfer_Cost'] = df_transfer_results['Total_Transfer_Cost'].round(2)
        print(f"  Total Units Transferred: {total_transferred_units:.0f}")
        print(f"  Total Transport Cost: {total_transport_cost:.2f}")
    else:
        print("  No transfers recommended.")

    unmet_summary = []
    for (s_to, p) in unmet_demand_vars:
        if unmet_demand_vars[(s_to, p)].varValue is not None and unmet_demand_vars[(s_to, p)].varValue > 0:
            unmet_summary.append({
                'Store_ID': s_to,
                'Product_Name': p,
                'Initial_Shortage': inventory_data_dict[(s_to,p)]['Initial_Shortage'],
                'Unmet_Demand_Units': unmet_demand_vars[(s_to, p)].varValue
            })
    df_unmet_summary = pd.DataFrame(unmet_summary)
    if not df_unmet_summary.empty:
        df_unmet_summary['Unmet_Demand_Units'] = df_unmet_summary['Unmet_Demand_Units'].astype(int)
        print("  Products with Remaining Unmet Demand.")
    else:
        print("  All initial shortages were met through transfers.")

    # --- 4. Visualization of Transfers ---
    fig = None
    if total_transferred_units > 0:
        df_transfer_summary = df_transfer_results.groupby(['From_Store', 'To_Store']).agg(
            Total_Units=('Transferred_Units', 'sum'),
            Total_Cost=('Total_Transfer_Cost', 'sum')
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_transfer_summary.sort_values(by='Total_Units', ascending=False).head(10),
                    x='From_Store', y='Total_Units', hue='To_Store', dodge=False, ax=ax)
        ax.set_title('Top 10 Store-to-Store Transfers by Units')
        ax.set_xlabel('From Store ID')
        ax.set_ylabel('Total Transferred Units')
        plt.xticks(rotation=45, ha='right')
        ax.legend(title='To Store ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

    return df_transfer_results, df_unmet_summary, fig

