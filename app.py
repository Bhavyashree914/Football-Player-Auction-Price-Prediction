import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Load Model
# ------------------------------
with open("auction_model.pkl", "rb") as f:
    model = pickle.load(f)
# ------------------------------
# App Layout
# ------------------------------
st.set_page_config(
    page_title="Football Player Auction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Prediction", "💰 Budget Simulation"])

# ------------------------------
# Home Page
# ------------------------------
if page == "🏠 Home":
    st.title(" Football Player Auction Price Prediction")
     
    st.markdown(
        """
        ### Welcome to the Player Auction Predictor!   

        
        """
    )

    st.image(
        "images/homepage.jpg",
        use_container_width=True,
    )

    st.success("👉 Start by selecting **Prediction** from the sidebar!")

# ------------------------------
# Prediction Page
# ------------------------------
elif page == "📊 Prediction":
    st.header("📊 Player Auction Price Prediction")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV file with player details", type="csv"
    )

    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)

        # Store names & positions for display
        display_info = (
            new_data[["Name", "Position"]]
            if "Name" in new_data.columns and "Position" in new_data.columns
            else pd.DataFrame()
        )

        # Ensure all required features are present
        required_features = model.feature_names_in_
        missing_features = [
            col for col in required_features if col not in new_data.columns
        ]

        if missing_features:
            st.error(f"Missing required features: {', '.join(missing_features)}")
        else:
            # Drop non-feature columns
            drop_cols = [
                col for col in ["Name", "Position", "AuctionPrice"] if col in new_data.columns
            ]
            X_new = new_data.drop(columns=drop_cols)

            # Make predictions
            predictions = model.predict(X_new)

            # Combine results with player names
            if not display_info.empty:
                results = display_info.copy()
                results["Predicted_AuctionPrice"] = predictions.round(2)
            else:
                results = pd.DataFrame(
                    {"Predicted_AuctionPrice": predictions.round(2)}
                )

            # Show results
            st.subheader("📋 Prediction Results")
            st.dataframe(results, use_container_width=True)

            # Single player view
            if "Name" in new_data.columns:
                selected_player = st.selectbox(
                    "🔍 View single player prediction",
                    results["Name"].tolist(),
                )
                player_row = results[results["Name"] == selected_player]
                st.metric(
                    label=selected_player,
                    value=f"${player_row['Predicted_AuctionPrice'].values[0]:,.2f}",
                )

            # Download predictions
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="predicted_auction_prices.csv",
                mime="text/csv",
            )

# ------------------------------
# Budget Simulation Page
# ------------------------------
elif page == "💰 Budget Simulation":
    st.header("💰 Team Budget Simulation")

    uploaded_file = st.file_uploader(
        "Upload the same CSV file with player details", type="csv", key="budget_upload"
    )

    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)

        # Drop non-feature columns for prediction
        drop_cols = [
            col for col in ["Name", "Position", "AuctionPrice"] if col in new_data.columns
        ]
        X_new = new_data.drop(columns=drop_cols)

        predictions = model.predict(X_new)

        results = new_data[["Name"]] if "Name" in new_data.columns else pd.DataFrame()
        results["Predicted_AuctionPrice"] = predictions.round(2)

        budget = st.number_input("Enter your total budget", min_value=0, value=300)

        if st.button("Run Budget Simulation"):
            sorted_players = results.sort_values(by="Predicted_AuctionPrice")
            selected = []
            remaining_budget = budget

            for _, player in sorted_players.iterrows():
                price = player["Predicted_AuctionPrice"]
                if price <= remaining_budget:
                    selected.append(player["Name"])
                    remaining_budget -= price

            if selected:
                st.success("✅ Players Selected")
                st.table(pd.DataFrame(selected, columns=["Player Name"]))
                st.info(f"💵 Remaining Budget: ${remaining_budget}")
            else:
                st.warning("⚠️ Budget too low to select any player.")
