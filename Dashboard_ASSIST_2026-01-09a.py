# -*- coding: utf-8 -*-
"""
OFI Analysis Dashboard
@author: Daniel
"""

import pandas as pd
import streamlit as st
from datetime import timedelta
from datetime import date
import plotly.express as px
import textwrap
import plotly.colors
import numpy as np

def generate_insight(comparison_df):
    """
    Generates insights using normalised daily averages and percentage changes.
    """
    # 1. Overall Totals
    total_prior_avg = comparison_df["Comparison Period (Daily Avg)"].sum()
    total_curr_avg = comparison_df["Filter Period (Daily Avg)"].sum()
    
    if total_prior_avg == 0:
        return "Insight: Comparison period had zero OFIs; percentage change cannot be calculated."

    net_change_val = total_curr_avg - total_prior_avg
    net_change_pct = (net_change_val / total_prior_avg) * 100
    
    # Separate increases and decreases
    increases = comparison_df[comparison_df["Change"] > 0].copy()
    decreases = comparison_df[comparison_df["Change"] < 0].copy()
    
    # Net change
    direction = "up" if net_change_val > 0 else "down"
    net_text = f"The overall daily average of OFIs is **{direction} {abs(net_change_pct):.1f}%** (from {total_prior_avg:.2f} to {total_curr_avg:.2f} per day)."

    # Identify top contributors to the increase
    contribution_text = ""
    if not increases.empty:
        # Calculate % increase for each specific classification
        # Handle division by zero for new classifications that didn't exist in prior period
        increases["Class_Pct_Change"] = (increases["Change"] / increases["Comparison Period (Daily Avg)"].replace(0, 0.001)) * 100
        
        top_inc = increases.sort_values("Change", ascending=False).head(2)
        
        inc_list = []
        for name, row in top_inc.iterrows():
            pct_label = f"{row['Class_Pct_Change']:.1f}%" if row['Comparison Period (Daily Avg)'] > 0 else "New"
            inc_list.append(f"**{name}** (+{pct_label})")
        
        contribution_text = f" Significant increases were seen in {', '.join(inc_list)}."

    # Identify offsetting decreases
    offset_text = ""
    if not decreases.empty:
        # Gross increase is the sum of all 'Change' values > 0
        gross_increase = increases["Change"].sum()
        total_reduction = abs(decreases["Change"].sum())
        
        if gross_increase > 0:
            offset_pct = (total_reduction / gross_increase) * 100
            
            # Find the classification with the biggest % drop
            best_improver = decreases.sort_values("Change", ascending=True).iloc[0]
            improver_name = best_improver.name
            improver_pct = (best_improver['Change'] / best_improver['Comparison Period (Daily Avg)']) * 100
            
            offset_text = (
                f" Reductions in other areas, led by **{improver_name}** ({abs(improver_pct):.1f}% drop), "
                f"offset **{min(offset_pct, 100):.1f}%** of the gross increase."
            )
        else:
            # If everything is a decrease
            top_drop = decreases.sort_values("Change", ascending=True).iloc[0]
            top_drop_pct = (top_drop['Change'] / top_drop['Comparison Period (Daily Avg)']) * 100
            offset_text = f" The largest improvement was in **{top_drop.name}**, which fell by **{abs(top_drop_pct):.1f}%**."

    return f"**Insight:** {net_text}{contribution_text}{offset_text}"


def wrap_client_names(names_list, width=100):
    """Joins names with commas and wraps the string at a specified width using <br>."""
    full_string = ", ".join(names_list)
    # textwrap.wrap returns a list of strings; we join them with HTML line breaks
    return "<br>".join(textwrap.wrap(full_string, width=width))


def load_data(ofi, issues, lob, loc):
    # Autopopulate dataframes with neccesary columns to allow merging to work without need for error exceptions
    ofi_df = pd.DataFrame(columns=['ClientID', 'Client','Date', 'CS Classification', 'CS Sub Classification', 'CS Note', 'OFI'])
    issues_df = pd.DataFrame(columns=['Client Id', 'Client','Date', 'Note', 'Reason'])
    loc_df = pd.DataFrame(columns=['ClientID']) 
    lob_df = pd.DataFrame(columns=['ClientID']) 

    # Use lists for concatenation
    if ofi: ofi_df = pd.concat([pd.read_csv(f, sep='\t') for f in ofi], ignore_index=True) 
    if issues: issues_df = pd.concat([pd.read_csv(f, sep='\t') for f in issues], ignore_index=True)
    
    # Only read optional files if they were uploaded
    if loc: loc_df = pd.read_csv(loc)
    if lob: lob_df = pd.read_csv(lob)
    
    # OFIS DATAFRAME PROCESSING ---------------------------    
    ofi_df = ofi_df.rename(columns={'CS Classification': 'Classification', 'CS Sub Classification':'Sub Classification'})
    
    # Only filter if ofi_df is not empty
    if not ofi_df.empty:
        ofi_df = ofi_df[
            ~ofi_df['Classification'].isin(['OFI FOR RECORDS', 'DUPLICATE', 'OFI IN ERROR', 'REPEATED OFI'])
        ]
        ofi_df['Date'] = pd.to_datetime(ofi_df['Date'], format='%d/%m/%Y %H:%M')
    
    # PROCESSING LOCATION & LOB DATAFRAMES ---------------------------
    loc_df = loc_df.rename(columns={'Client ID': 'ClientID'}).drop(columns=['Client'], errors='ignore')
    lob_df = lob_df.rename(columns={'Id': 'ClientID', 'Classification':'Revenue'}).drop(columns=['Client'], errors='ignore')
    
    # PROCESSING ISSUES DATAFRAME ---------------------------  
    issues_df = issues_df.rename(columns={'Client Id': 'ClientID', 'Note':'CS Note'}).drop(columns=[], errors='ignore')
      
    reason_to_classification = {
        'Site Left Unsecured':'COMPLAINT',
        'Missed Collection':'COMPLAINT',
        'Collection Time Complaint':'COMPLAINT',
        'Complaint':'COMPLAINT',
        'Accident/Incident':'COMPLAINT', 
        'Dirty/Broken Bin':'COMPLAINT',
        'Missing Bin':'COMPLAINT', 
        'Billing Issue':'COMPLAINT',
    }
    
    reason_to_sub_classification = {
        'Site Left Unsecured':'Site Left Unsecured',
        'Missed Collection':'Missed Collection Complaint',
        'Collection Time Complaint':'Collection Time Complaint',
        'Complaint':'General Client Complaint',
        'Accident/Incident':'Accident/Incident Complaint', 
        'Dirty/Broken Bin':'Dirty/Broken Bin Complaint',
        'Missing Bin':'Missing Bin Complaint', 
        'Billing Issue':'Billing Issue Complaint'
    }
    
    issues_df['Classification'] = issues_df['Reason'].map(reason_to_classification)
    issues_df['Sub Classification'] = issues_df['Reason'].map(reason_to_sub_classification)
    
    if not issues_df.empty:
        issues_df['Date'] = pd.to_datetime(issues_df['Date'], format='%d/%m/%Y %H:%M:%S')
    
    # MERGING ---------------------------
    ofi_issue_df = ofi_df.merge(
        issues_df, 
        on=['ClientID', 'Client','Date', 'CS Note', 'Classification', 'Sub Classification'], 
        how='outer'
    )
    
    df = (
        ofi_issue_df
        .merge(loc_df, on='ClientID', how='left')
        .merge(lob_df, on='ClientID', how='left')
    )
    
    df.loc[df['ClientID'] == 7769, 'Client'] = 'Public Complaints/Feedback'
    
    # Ensure columns exist before string operations
    if 'Sub Classification' in df.columns:
        df['Sub Classification'] = (
            df['Sub Classification']
            .fillna(df.get('Classification', ''))
            .astype(str)
            .str.lower()
        )
    
    df = df.drop_duplicates(subset=['Date', 'ClientID', 'Classification', 'OFI'])

    return df


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="OFI/Issue Dashboard",
    page_icon="🔨",
    layout="wide"
)

st.title("OFI/Issue Dashboard")

# --------------------------------------------------
# FILE UPLOADS
# --------------------------------------------------
with st.sidebar.expander("Upload files", expanded=True):

    ofi_files = st.file_uploader("OFI export (.tsv)", type=["tsv"], accept_multiple_files=True)
    issue_files = st.file_uploader("Issues export (.tsv)", type=["tsv"], accept_multiple_files=True)
    loc_file = st.file_uploader("Locations file (.csv)", type=["csv"])
    lob_file = st.file_uploader("LoB & BC file (.csv)", type=["csv"])
    
    if st.button("Process Uploaded Files"):
        st.session_state.df = load_data(ofi_files, issue_files, lob_file, loc_file)
        st.success("Data processed!")

if not (ofi_files or issue_files):
    st.info("For this programme to continue, you are required to upload at least one: OFI file, or Issues file.")
    st.stop()

# --------------------------------------------------
# DATA LOADING & PRE-PROCESSING
# --------------------------------------------------

# Access processed data
if "df" not in st.session_state:
    st.warning("Click 'Process Uploaded Files' to start")
    st.stop()

df = st.session_state.df

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
with st.sidebar.form("filters"):
    st.header("Filters")
    
    # filters only show if the file was uploaded
    if 'LOB' in df.columns:
        lobs = sorted(df['LOB'].dropna().unique())
        selected_lobs = st.multiselect("LOB", lobs, default=lobs)

    if 'Revenue' in df.columns:
        revenues = sorted(df['Revenue'].dropna().unique())
        selected_revenue = st.multiselect("Revenue Tier", revenues, default=revenues)

    min_date = df['Date'].min().date()
    max_date = date.today()

    start_date = None
    end_date = None
    date_range = st.date_input(
        "Date range",
        value=[],
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY"
    )
        
    if len(date_range) == 2:
        start_date, end_date = date_range

    submitted = st.form_submit_button("Apply Filters")

    if submitted:
        if start_date is None or end_date is None:
            st.warning("Please select a valid date range .")
        else:
            st.success(f"Filtering from {start_date} to {end_date}")

# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------
filtered_nodate = df

# --------------------------------------------------
# CONDITIONAL LOB / REVENUE FILTERING
# --------------------------------------------------

if 'LOB' in df.columns:
    if set(selected_lobs) != set(lobs):
        filtered_nodate = filtered_nodate[filtered_nodate["LOB"].isin(selected_lobs)]

if 'Revenue' in df.columns:
    if set(selected_revenue) != set(revenues):
        filtered_nodate = filtered_nodate[filtered_nodate["Revenue"].isin(selected_revenue)]
    
# --------------------------------------------------
# Date Filtering
# --------------------------------------------------
# Convert date_input outputs to timestamps
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

filtered = filtered_nodate[
    (filtered_nodate["Date"] >= start_ts) &
    (filtered_nodate["Date"] < end_ts)
]

# --------------------------------------------------
# KPI ROW 
# --------------------------------------------------

# Calculate the date ranges based on the filtered dataframe
if not filtered.empty:
    latest_date = filtered["Date"].max()
    earliest_date = filtered["Date"].min()
    total_days = (latest_date - earliest_date).days + 1
    
    # Split the filtered period into two equal halves
    half_days = total_days // 2
    mid_point = earliest_date + pd.Timedelta(days=half_days)

    # Current period (2nd half) and Previous period (1st half)
    curr_period = filtered[filtered["Date"] >= mid_point]
    prev_period = filtered[filtered["Date"] < mid_point]
    
    # Calculate days in each for dn/dt normalization
    days_curr = (latest_date - mid_point).days + 1
    days_prev = (mid_point - earliest_date).days
else:
    curr_period = prev_period = filtered
    total_days = days_curr = days_prev = 0

# Movement calc
prev_counts = prev_period["Classification"].value_counts().rename("Previous")
curr_counts = curr_period["Classification"].value_counts().rename("Current")

movement = pd.concat([prev_counts, curr_counts], axis=1).fillna(0)

# dn/dt calculation 
movement["dn_dt"] = (movement["Current"] / days_curr) - (movement["Previous"] / days_prev) if days_prev > 0 else 0
movement["Pct Change"] = ((movement["Current"] - movement["Previous"]) / movement["Previous"].replace(0, np.nan)) * 100

# Identifies biggest change using dn/dt
MIN_PRIOR_COUNT = 5 
eligible = movement[movement["Previous"] >= MIN_PRIOR_COUNT].copy()

top_increase = eligible.sort_values("dn_dt", ascending=False).head(1)
top_decrease = eligible.sort_values("dn_dt", ascending=True).head(1)

# Helper for metric formatting
def format_delta_stat(row):
    if row.empty: return None, None
    name = row.index[0]
    pct = row["Pct Change"].iloc[0]
    # Displays the Pct Change but it's sorted by the daily rate (dn/dt)
    return name, f"{pct:+.1f}% vs prior period"

inc_name, inc_delta = format_delta_stat(top_increase)
dec_name, dec_delta = format_delta_stat(top_decrease)

# Patreo analysis  (on the full filtered selection)
if not filtered.empty:
    client_counts = filtered["ClientID"].value_counts().reset_index()
    client_counts.columns = ["ClientID", "Count"]
    client_counts["CumPct"] = client_counts["Count"].cumsum() / client_counts["Count"].sum()
    pareto_clients = (client_counts["CumPct"] >= 0.5).idxmax() + 1
else:
    pareto_clients = 0

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

k1, k2, k3 = st.columns(3)
k1.metric("Total OFIs", len(filtered), border=True, help=f"Total in selected period ({total_days} days)")
k2.metric("Active Clients", filtered["ClientID"].nunique(), border=True)
k3.metric("Top 50% OFI Concentration", f"{pareto_clients} Clients", border=True, help="Measures issue concentration: identifies the few sites where 50% of OFIs/issues are generated")

k4, k5, k6 = st.columns(3)
k4.metric("Top Category", filtered["Classification"].mode()[0] if not filtered.empty else "N/A", border=True, height='stretch')


k5.metric("Greatest Increasing Trend", inc_name or "N/A", delta=inc_delta, delta_color="inverse", border=True, 
          help=f"Highest daily rate of increase ($dn/dt$) comparing the last {days_curr} days to the preceding {days_prev} days.")

k6.metric("Greatest Decreasing Trend", dec_name or "N/A", delta=dec_delta, delta_color="inverse", border=True,
          help=f"Highest daily rate of decrease ($dn/dt$) comparing the last {days_curr} days to the preceding {days_prev} days.")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_class, tab_trends, tab_movers, tab_geo, tab_customer = st.tabs(
    ["Classification", "Trends", "Variations", "Geography", "Client"]
)

# -----------------------------------------------------------------------------
# GEOGRAPHY TAB
# -----------------------------------------------------------------------------
with tab_geo:
    st.header("OFI Geographic Distribution")
    if 'Latitude' in df.columns:
        with st.container(border=True):
                
            map_df = (
                filtered
                .groupby(['ClientID', 'Client', 'Latitude', 'Longitude'])
                .size()
                .reset_index(name='Total_OFIs')
            )
        
            fig_map = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                size="Total_OFIs",
                color="Total_OFIs",
                color_continuous_scale="Viridis",
                hover_name="Client",
                zoom=9,
                mapbox_style="carto-positron",
                height=600
            )
            
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                dragmode='pan'
            )
                    
            st.plotly_chart(fig_map, width='stretch')
    else:
        st.info('Geographic distribution is not visible, please upload a location file if you wish to generate this display')

# -----------------------------------------------------------------------------
# TRENDS TAB
# -----------------------------------------------------------------------------

with tab_trends:        
    st.header("Trends")
    
    # -------------------------------------------------------------------------
    # TRENDS LINE GRAPH 
    # -------------------------------------------------------------------------
    with st.container(border=True):
        
        st.subheader("Historical Trends")
        
        with st.container(border=True):
        
            granularity_options = ["Daily", "Weekly", "Fortnightly", "Monthly", "Quarterly", "6-Monthly", "Annual"]
            granularity = st.select_slider(
                "Select Trend Granularity",
                options=granularity_options,
                value="Quarterly",
                key="trends_granularity"
            )
            
            freq_map = {
                "Daily": "D", "Weekly": "W", "Fortnightly": "2W", 
                "Monthly": "ME", "Quarterly": "QE", "6-Monthly": "2QE", "Annual": "YE"
            }
            freq = freq_map[granularity]
        
            view_mode = st.radio(
                "View mode", ["Total", "By Classification", "By Sub-Classification"],
                horizontal=True, key="trends_view_mode"
            )
    
            base = filtered_nodate.set_index("Date")
            if view_mode == "Total":
                trend = base.resample(freq).size().reset_index(name="OFI Count")
                fig_line = px.line(trend, x="Date", y="OFI Count", markers=True, title="Total OFIs Over Time")
            elif view_mode == "By Classification":
                trend = base.groupby("Classification").resample(freq).size().reset_index(name="OFI Count")
                fig_line = px.line(trend, x="Date", y="OFI Count", color="Classification", markers=True)
            else:
                focus_class = st.selectbox("Select classification", sorted(filtered_nodate["Classification"].unique()))
                sub_data = base[base["Classification"] == focus_class]
                trend = sub_data.groupby("Sub Classification").resample(freq).size().reset_index(name="OFI Count")
                fig_line = px.line(trend, x="Date", y="OFI Count", color="Sub Classification", markers=True)
        
        
        fig_line.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            dragmode='pan'
        )
        
        
        st.plotly_chart(fig_line, width='stretch')

    # -------------------------------------------------------------------------
    # NORMALISED WATERFALL CONTRIBUTION
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("Contribution to OFI Change", help="This diagram displays the contribution of each classification to the change in daily average OFIs.")
        
        # Standard date logic
        filter_start = filtered["Date"].min()
        filter_end = filtered["Date"].max()
        n_days_filter = (filter_end - filter_start).days + 1
        
        with st.container(border=True):

                
            def_comparison_end = filter_start - timedelta(days=1)
            try:
                def_comparison_start = max((def_comparison_end - timedelta(days=n_days_filter - 1)).date(), filtered_nodate['Date'].min().date())
            except:
                st.toast("Please ensure that your uploaded file contains entires within your selected date range in the sidebar",icon='spinner',duration=6)
                st.stop()
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.date_input(
                        "Filtered Date Period",
                        value=[filter_start, filter_end],
                        disabled=True,
                        label_visibility="visible",
                        format="DD/MM/YYYY",
                        key="wf_filter_range",
                    )
                except:
                    st.toast("No OFI or Issue entries match filters.")
                    st.stop() 
                    
            with col2:
                comparison_range = st.date_input(
                    "Select Comparison Period",
                    value=[def_comparison_start, def_comparison_end],
                    max_value = filter_start - timedelta(days=1),
                    min_value = filtered_nodate['Date'].min().date(),
                    key="wf_comparison_range",
                    format="DD/MM/YYYY"
                )
                
            col1, col2 = st.columns(2)
            with col1:
                view_level = st.radio(
                    "Select view mode:",
                    options=["Classification", "Sub-Classification"],
                    horizontal=True,
                    key="wf_view_level"
                )
            with col2:
                metric_choice = st.radio(
                    "Select metric type:",
                    options=["Absolute Change", "Percentage Change"],
                    horizontal=True,
                    key="wf_metric_type"
                )

                use_percent = (metric_choice == "Percentage Change")
        
        group_col = "Classification" if view_level == "Classification" else "Sub Classification"
    
        if len(comparison_range) == 2:
            comparison_start, comparison_end = comparison_range
            n_days_comparison = (comparison_end - comparison_start).days + 1
            
            df_filter_period = filtered_nodate[(filtered_nodate["Date"] >= pd.Timestamp(filter_start)) & (filtered_nodate["Date"] <= pd.Timestamp(filter_end))]
            df_comparison_period = filtered_nodate[(filtered_nodate["Date"] >= pd.Timestamp(comparison_start)) & (filtered_nodate["Date"] <= pd.Timestamp(comparison_end))]
            
            comparison_df = pd.DataFrame(index=sorted(filtered_nodate[group_col].unique()))
            comparison_df["Filter Period (Daily Avg)"] = (comparison_df.index.map(df_filter_period.groupby(group_col).size()).fillna(0)) / n_days_filter
            comparison_df["Comparison Period (Daily Avg)"] = (comparison_df.index.map(df_comparison_period.groupby(group_col).size()).fillna(0)) / n_days_comparison
            
            before = comparison_df["Comparison Period (Daily Avg)"]
            after = comparison_df["Filter Period (Daily Avg)"]
            
            comparison_df["Change"] = (after - before).round(2)
            y_label, hover_fmt, suffix = "Change (Avg OFIs/Day)", ":.2f", ""
    
            st.info(generate_insight(comparison_df))

    
            if use_percent:
                # 0 if both are 0; 100 if only before is 0; else default
                comparison_df["Change"] = np.where(
                    (before == 0) & (after == 0), 0,
                    np.where((before == 0) & (after > 0), 100, ((after - before) / before.replace(0, np.nan)) * 100)
                ).round(1)
                y_label, hover_fmt, suffix = "% Change in Avg OFIs", ":.1f", "%"
            
            waterfall_df = comparison_df[comparison_df["Change"] != 0].sort_values("Change", ascending=False).reset_index().rename(columns={"index": group_col})
    
            if not waterfall_df.empty:
                waterfall_df["Direction"] = waterfall_df["Change"].apply(lambda x: "Increase" if x > 0 else "Decrease")
                
                fig_waterfall = px.bar(
                    waterfall_df, x=group_col, y="Change", color="Direction",
                    color_discrete_map={"Increase": "#EF5350", "Decrease": "#66BB6A"},
                    title=f"{metric_choice} in Daily Avg OFIs",
                    labels={"Change": y_label, "Comparison Period (Daily Avg)": "Before Avg", "Filter Period (Daily Avg)": "After Avg"},
                    hover_data={"Change": f"{hover_fmt}{suffix}", "Comparison Period (Daily Avg)": ":.2f", "Filter Period (Daily Avg)": ":.2f", "Direction": False},
                    text_auto=f"{hover_fmt}{suffix}"
                )
                fig_waterfall.update_layout(margin=dict(l=0, r=0, t=30, b=0), dragmode='pan')
                fig_waterfall.add_hline(y=0, line_dash="dash", line_color="black")
                
                st.plotly_chart(fig_waterfall, width='stretch')
            else:
                st.caption(f"No significant change in daily averages for {view_level} between these periods.")

# -----------------------------
# MOVERS TAB
# -----------------------------

with tab_movers:
    st.header("OFI Variations")
    
    with st.container(border=True):
        
        with st.container(border=True, height='stretch'):
            
            # Find and display filter period date range, the data from this range is then compared to the comparison period date range
            filter_start = filtered["Date"].min()
            filter_end = filtered["Date"].max()
            n_days_filter = (filter_end - filter_start).days + 1
            
            col1,col2 = st.columns(2)
            with col1:
                st.date_input(
                    "Filtered Date Period",
                    value=[filter_start, filter_end],
                    disabled=True,
                    label_visibility="visible",
                    format="DD/MM/YYYY",
                    key = 'movers_filter_range_quad'
                )  
             
            with col2:
                def_comparison_end = filter_start - timedelta(days=1)
                def_comparison_start = max((def_comparison_end - timedelta(days=n_days_filter - 1)).date(), filtered_nodate['Date'].min().date())
    
                comparison_range = st.date_input(
                    "Select Comparison Period",
                    value=[def_comparison_start, def_comparison_end],
                    max_value=filter_start - timedelta(days=1),
                    min_value=filtered_nodate['Date'].min().date(),
                    key="movers_comparison_range_quad",
                    format="DD/MM/YYYY"
                )
    
            # Ensure comparison_range is valid before defining movement
            if len(comparison_range) == 2:
                comparison_start, comparison_end = comparison_range
                n_days_comparison = (comparison_end - comparison_start).days + 1
                
        df_filter_period = filtered_nodate[(filtered_nodate["Date"] >= pd.Timestamp(filter_start)) & (filtered_nodate["Date"] <= pd.Timestamp(filter_end))]
        df_comparison_period_period = filtered_nodate[(filtered_nodate["Date"] >= pd.Timestamp(comparison_start)) & (filtered_nodate["Date"] <= pd.Timestamp(comparison_end))]
        
        # Raw counts and averages per client
        filter_period_counts = df_filter_period.groupby("Client").size().reset_index(name="Filter Period Raw Count")
        filter_period_counts["Filter Period Avg OFIs/Day"] = filter_period_counts["Filter Period Raw Count"] / n_days_filter
        
        comparison_period_counts = df_comparison_period_period.groupby("Client").size().reset_index(name="Comparison Period Raw Count")
        comparison_period_counts["Comparison Period Avg OFIs/Day"] = comparison_period_counts["Comparison Period Raw Count"] / n_days_comparison
        
        # Merge into the movement dataframe
        movement = filter_period_counts.merge(comparison_period_counts, on="Client", how="outer").fillna(0)
        movement["Change (Δ Avg)"] = movement["Filter Period Avg OFIs/Day"] - movement["Comparison Period Avg OFIs/Day"]
        
        
        # Group by coordinates to find clients that overlap exactly
        grouped = movement.groupby(["Comparison Period Avg OFIs/Day", "Filter Period Avg OFIs/Day"]).agg({
            "Client": lambda x: wrap_client_names(x, width=50),
            "Filter Period Raw Count": "sum",
            "Comparison Period Raw Count": "sum",
            "Change (Δ Avg)": "first"
        }).reset_index()
        
        # Define the Trend for the group
        grouped["Trend"] = grouped["Change (Δ Avg)"].apply(
            lambda x: "Increasing" if x > 0.0001 else ("Decreasing" if x < -0.0001 else "No Change"))
                        
        max_val = grouped[["Comparison Period Avg OFIs/Day", "Filter Period Avg OFIs/Day"]].max().max() * 1.1

        # Scatterplot 
        
        st.subheader("Period-on-Period Comparison of OFIs by Client",help='This scatterplot displays the changes in the rate of OFIs for each client showing the rate in the selected Current Period vs the selected Comparison period. Dots above the line dneote client(s) whose OFI rate have increased in the current period relative to the Comparison whilst dots below the line denote clients whose OFIs have decreased in the Current period relative to the Comparison Period')

        fig_scatter = px.scatter(
            grouped,
            x="Comparison Period Avg OFIs/Day",
            y="Filter Period Avg OFIs/Day",
            color="Trend",
            size="Filter Period Raw Count", 
            hover_name="Client", 
            hover_data={
                "Comparison Period Avg OFIs/Day": ":.2f",
                "Filter Period Avg OFIs/Day": ":.2f",
                "Filter Period Raw Count": True,
                "Trend": False
            },
            opacity=0.7,
            title="Client OFI Frequency",
            color_discrete_map={'Increasing': '#EF553B', 'Decreasing': '#00CC96', 'No Change': '#636EFA'},
            height=650
        )
        
        fig_scatter.update_layout(
            margin=dict(l=0, r=0, t=25, b=0),
            dragmode='pan'
        )
                        
        # 45-degree reference line
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="black", width=1, dash="dot"))
        
        st.plotly_chart(fig_scatter, width='stretch')

        # -----------------------------------------------------------------
        # KEY MOVERS TABLE
        # -----------------------------------------------------------------
        st.divider()
        st.subheader("Significant Client Fluctuations", help="This table displays all clients that had a decrease or increase of more than ± 0.05 OFIs/day between the selected Current Period and selected Comparison Period")
        
        # Filter for clients with a notable change
        significant_movers = movement[movement["Change (Δ Avg)"].abs() > 0.05].sort_values("Change (Δ Avg)", ascending=False)

        if not significant_movers.empty:
            # Create the base dataframe
            display_df = significant_movers[["Client", "Comparison Period Raw Count", "Filter Period Raw Count", "Change (Δ Avg)"]].copy()
            
            # Calculate % Change 
            display_df["% Change"] = ((display_df["Filter Period Raw Count"] - display_df["Comparison Period Raw Count"]) / display_df["Comparison Period Raw Count"].replace(0, 1)) * 100
            
            # Round decimals and apply the color gradient 
            styled_df = (
                display_df.style
                .background_gradient(subset=["Change (Δ Avg)"], cmap="RdYlGn_r")
                .format({
                    "Change (Δ Avg)": "{:.2f}",
                    "% Change": "{:.1f}%",
                    "Comparison Period Raw Count": "{:.0f}",
                    "Filter Period Raw Count": "{:.0f}"
                })
            )
 
            st.dataframe(
                styled_df,
                width='stretch',
                hide_index=True
            )
        
        else:
            st.info("No significant movers found for these periods.")
                
# --------------------------------------------------
# CLASSIFICATION TAB
# --------------------------------------------------
with tab_class:
        
    st.header("OFI Classifications")

    # ---------- MAIN CLASSIFICATION ----------
    with st.container(border=True):
    
        class_counts = (
            filtered["Classification"]
            .value_counts()
            .reset_index()
        )
        class_counts.columns = ["Classification", "Count"]
        class_counts["Percent"] = (
            class_counts["Count"] / class_counts["Count"].sum() * 100
        ).round(1)
    
        fig_class = px.bar(
            class_counts,
            x="Count",
            y="Classification",
            orientation="h",
            color="Count",
            hover_data={
                "Count": True,
                "Percent": ":.1f"
            },
            title="OFIs by Classification"
        )
                    
        fig_class.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            dragmode='pan')

        st.plotly_chart(fig_class, width='stretch')

    # ---------- SUB-CLASSIFICATION DRILL ----------
    with st.container(border=True):

        focus_class = st.selectbox(
            "Look further into sub-classification",
            class_counts["Classification"].tolist()
        )
    
    
        sub = filtered[filtered["Classification"] == focus_class]
    
        sub_counts = (
            sub["Sub Classification"]
            .value_counts()
            .reset_index()
        )
        sub_counts.columns = ["Sub Classification", "Count"]
        sub_counts["Percent"] = (
            sub_counts["Count"] / sub_counts["Count"].sum() * 100
        ).round(1)
    
        fig_sub = px.bar(
            sub_counts,
            x="Count",
            y="Sub Classification",
            orientation="h",
            color="Count",
            hover_data={
                "Count": True,
                "Percent": ":.1f"
            },
            title=f"{focus_class} – Sub-Classification Breakdown"
        )
                
        fig_sub.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            dragmode='pan')
    
        st.plotly_chart(fig_sub, width='stretch')


# --------------------------------------------------
# CUSTOMER TAB
# --------------------------------------------------
with tab_customer:
    st.header("OFIs by Client")
    
    with st.container(border=True):
        client_counts = (
            filtered
            .groupby(["ClientID", "Client"])
            .size()
            .reset_index(name="OFI Count")
            .sort_values("OFI Count", ascending=False)
        )
    
        client_counts["label"] = (
            client_counts["Client"] +
            " (" + client_counts["OFI Count"].astype(str) + ")"
        )
        
        selected_label = st.selectbox(
            "Select Client (sorted by OFIs)",
            client_counts["label"]
        )
    
        selected_client = client_counts[
            client_counts["label"] == selected_label
        ].iloc[0]
    
        client_df = filtered[
            filtered["ClientID"] == selected_client["ClientID"]
        ]
    
        c1, c2 = st.columns(2)
        c1.metric("Total OFIs", len(client_df), border=True)
        c2.metric("Unique Issues", client_df["Classification"].nunique(), border=True)
    
        with st.container(border=True):
            
            # Aggregate
            breakdown = (
                client_df
                .groupby(["Classification", "Sub Classification"])
                .size()
                .reset_index(name="Count")
            )
            
            # Sort classifications by total count
            class_order = (
                breakdown
                .groupby("Classification")["Count"]
                .sum()
                .sort_values(ascending=True)
                .index
            )
            
            breakdown["Classification"] = pd.Categorical(
                breakdown["Classification"],
                categories=class_order,
                ordered=True
            )
            
            breakdown = breakdown.sort_values(
                ["Classification", "Count"],
                ascending=[True, False]
            )
            
            color_map = {}
            
            for i, classification in enumerate(class_order):
                subs = breakdown.loc[
                    breakdown["Classification"] == classification,
                    "Sub Classification"
                ].unique()
                
                n = len(subs)
                
                if n == 1:
                    # Pick the midpoint if there is only one sub-classification
                    shades = plotly.colors.sample_colorscale("plasma", [0.5])
                else:
                    sample_points = [j / (n - 1) for j in range(n)]
                    shades = plotly.colors.sample_colorscale("plasma", sample_points)
            
                for sub, shade in zip(subs, shades):
                    color_map[sub] = shade
            
            fig_client = px.bar(
                breakdown,
                x="Count",
                y="Classification",
                color="Sub Classification",
                orientation="h",
                title="Issue Breakdown by Classification and Sub-Classification",
                color_discrete_map=color_map,
                height = 550
            )
            
            fig_client.update_layout(
                barmode="stack",
                yaxis_title="Classification",
                xaxis_title="OFI Count",
                margin=dict(l=0, r=0, t=30, b=0),
                dragmode='pan'
            )
            
            for trace in fig_client.data:
                classification = breakdown.loc[
                    breakdown["Sub Classification"] == trace.name,
                    "Classification"
                ].iloc[0]
            
                trace.legendgroup = classification
                trace.legendgrouptitle = dict(text=classification)
        
            st.plotly_chart(fig_client, width='stretch')
            
        with st.container(border=True):
            st.dataframe(client_df, width='stretch')


# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
with st.expander("Data Preview"):
    st.dataframe(filtered, width='stretch')

