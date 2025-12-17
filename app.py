
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.model_selection


# Page configuration
st.set_page_config(
    page_title="F1 Race Winners Dashboard",
    layout="wide"
)

st.title("🏎️ Formula 1 Race Winners Dashboard")
st.write("Simple interactive dashboard built using Streamlit")

# Load dataset
file_path = "Taruni123.csv.xlsx"
df = pd.read_excel(file_path)

# Show basic info
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.subheader("📊 Dataset Summary")
st.write(f"Total Records: {df.shape[0]}")
st.write(f"Total Columns: {df.shape[1]}")



# ------------------ BLOCK 2 : SIDEBAR FILTERS ------------------

st.sidebar.header("🔍 Filter Options")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Convert date column to datetime (safe)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Extract year
df["year"] = df["date"].dt.year


# Sidebar selections
selected_year = st.sidebar.multiselect(
    "Select Year(s)",
    options=sorted(df["year"].unique()),
    default=sorted(df["year"].unique())
)

selected_winner = st.sidebar.multiselect(
    "Select Winner(s)",
    options=sorted(df["winner"].unique()),
    default=sorted(df["winner"].unique())
)

selected_car = st.sidebar.multiselect(
    "Select Car(s)",
    options=sorted(df["car"].unique()),
    default=sorted(df["car"].unique())
)

# Apply filters
filtered_df = df[
    (df["year"].isin(selected_year)) &
    (df["winner"].isin(selected_winner)) &
    (df["car"].isin(selected_car))
]

st.subheader("📌 Filtered Data")
st.dataframe(filtered_df)


# ------------------ BLOCK 3 : BAR CHART - TOP DRIVERS ------------------

st.subheader("🏆 Top Drivers by Number of Wins")

# Count wins per driver (after filters)
driver_wins = filtered_df["winner"].value_counts().head(10)

# Plot
fig, ax = plt.subplots()
driver_wins.plot(kind="bar", ax=ax)

ax.set_xlabel("Driver")
ax.set_ylabel("Number of Wins")
ax.set_title("Top 10 Drivers by Wins")
plt.xticks(rotation=45)

st.pyplot(fig)


# =========================
# BLOCK 4 – TOP 10 CARS BY WINS
# =========================

st.subheader("🏎️ Top 10 Cars by Number of Wins")

# Count wins per car
car_wins = filtered_df["car"].value_counts().head(10)

# Plot
fig, ax = plt.subplots()
car_wins.plot(kind="bar", ax=ax)

ax.set_xlabel("Car")
ax.set_ylabel("Number of Wins")
ax.set_title("Top 10 Cars by Wins")
plt.xticks(rotation=45)

st.pyplot(fig)


# =========================
# BLOCK 5 – YEAR-WISE NUMBER OF RACES
# =========================

st.subheader("📈 Year-wise Number of F1 Races")

# Extract year from date (safe conversion)
df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

# Count races per year
races_per_year = df.groupby("year").size()

# Plot
fig, ax = plt.subplots()
ax.plot(races_per_year.index, races_per_year.values, marker="o")

ax.set_xlabel("Year")
ax.set_ylabel("Number of Races")
ax.set_title("Number of F1 Races per Year")

st.pyplot(fig)


# =========================
# BLOCK 6 – YEAR-WISE WINS OF TOP 5 DRIVERS
# =========================

st.subheader("🏆 Year-wise Wins of Top 5 Drivers")

# Find top 5 drivers overall
top_5_drivers = df["winner"].value_counts().head(5).index

fig, ax = plt.subplots()

for driver in top_5_drivers:
    yearly_wins = (
        df[df["winner"] == driver]
        .groupby("year")
        .size()
    )
    ax.plot(yearly_wins.index, yearly_wins.values, marker="o", label=driver)

ax.set_xlabel("Year")
ax.set_ylabel("Number of Wins")
ax.set_title("Top 5 Drivers – Wins Trend Over Years")
ax.legend()

st.pyplot(fig)


# =========================
# BLOCK 7 – FINAL NOTES & FOOTER
# =========================

st.markdown("---")

st.subheader("📌 Dashboard Summary")

st.write("""
• This dashboard analyzes historical Formula 1 race winner data  
• Interactive filters allow analysis by year, driver, and car  
• Visualizations help identify dominant drivers and constructors  
• Trend charts show how Formula 1 has evolved over time  
""")

st.markdown("📊 **Built using Python, Pandas, Matplotlib, and Streamlit**")



# =========================
# BLOCK 8 – SIMPLE ML TAB
# =========================

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.markdown("---")
st.subheader("🤖 Simple ML Model – Win Prediction")

st.write("This model predicts the number of wins for a selected driver based on year.")

# Select driver
selected_driver = st.selectbox(
    "Select a Driver for ML Prediction",
    df["winner"].unique()
)

# Prepare data for ML
ml_df = df[df["winner"] == selected_driver].copy()
ml_df["year"] = pd.to_datetime(ml_df["date"]).dt.year

# Count wins per year
wins_per_year = ml_df.groupby("year").size().reset_index(name="wins")

# Check minimum data
if len(wins_per_year) >= 5:
    X = wins_per_year[["year"]]
    y = wins_per_year["wins"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # User input year
    input_year = st.number_input(
        "Enter a year to predict wins",
        min_value=int(X["year"].min()),
        max_value=2030,
        value=int(X["year"].max())
    )

    prediction = model.predict([[input_year]])

    st.success(
        f"Predicted wins for {selected_driver} in {input_year}: {max(0, int(prediction[0]))}"
    )

else:
    st.warning("Not enough data for this driver to train an ML model.")
