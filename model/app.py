# app.py
import streamlit as st
import pandas as pd
import joblib


# ================================
# Load Trained Model
# ================================
@st.cache_resource
def load_model():
    return joblib.load('final_model.pkl')

model = load_model()

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="Terror Attack Success Predictor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Sidebar - Documentation
# ================================
with st.sidebar:
    st.markdown("# Model Overview")
    st.markdown("""
    ### **Terror Attack Success Prediction Model (TASPM)**  
    This model estimates the probability of a planned terrorist attack being successful, using machine learning trained on historical global terrorism data.

    **Objective:**  
    Identify patterns, predict operational success, and assist intelligence and counterterrorism strategies.

    **Key Benefits:**  
    - Early detection of high-risk incidents  
    - Improved situational awareness for analysts  
    - Data-driven decision support for law enforcement  

    **Model Inputs Include:**  
    - Attack year, country, and region  
    - Target and weapon details  
    - Attack type and motivation criteria  
    """)

    st.markdown("---")
    st.caption("Developed as part of the **Global and Egypt Terrorism Analytics** Project.")

# ================================
# App Title & Description
# ================================
st.markdown("<h1 style='text-align:center; color:#FF4B4B;'>Terror Attack Success Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Empowering intelligence through data — predict, analyze, and uncover the factors behind terrorist attack success</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ================================
# Input Fields
# ================================
st.header("Enter Attack Details")
st.caption("Please provide as much accurate information as possible. \nFields marked as 'Unknown' can be manually specified if necessary.")
col1, col2 = st.columns(2)

# ---------------- Country & Province ----------------
countries = [
    'Pakistan', 'Mexico', 'El Salvador', 'Turkey', 'Afghanistan', 'Somalia',
    'Yemen', 'Ukraine', 'Nigeria', 'India', 'Macedonia', 'Philippines', 'Iraq',
    'Cambodia', 'Burundi', 'Saudi Arabia', 'Nicaragua', 'Cameroon', 'Sri Lanka',
    'France', 'West Bank and Gaza Strip', 'Paraguay', 'Italy', 'Lebanon',
    'Nepal', 'East Germany (GDR)', 'Venezuela', 'Thailand', 'Ecuador', 'Kenya',
    'Mozambique', 'South Africa', 'United States', 'Spain', 'Syria', 'Tajikistan',
    'Algeria', 'Haiti', 'West Germany (FRG)', 'Jordan', 'Russia', 'Egypt',
    'Uganda', 'Peru', 'United Kingdom', 'Niger', 'Colombia', 'Sudan',
    'Democratic Republic of the Congo', 'Iran', 'Maldives', 'Bangladesh',
    'Austria', 'Libya', 'Greece', 'South Korea', 'Ireland', 'Chile', 'Argentina',
    'Israel', 'South Sudan', 'Myanmar', 'Japan', 'Mali', 'Portugal', 'Angola',
    'Honduras', 'Denmark', 'Kosovo', 'Georgia', 'Cuba', 'Kyrgyzstan', 'Germany',
    'Burkina Faso', 'Australia', 'Bolivia', 'Western Sahara',
    'Central African Republic', 'Guatemala', 'Liberia', 'Indonesia', 'Switzerland',
    'Belgium', 'Jamaica', 'Canada', 'Taiwan', 'Senegal', 'Dominican Republic',
    'Brazil', 'Serbia-Montenegro', 'Cyprus', 'Tanzania', 'Ethiopia',
    'United Arab Emirates', 'Romania', 'Zimbabwe', 'Laos', 'Armenia', 'Panama',
    'Slovak Republic', 'Yugoslavia', 'Rwanda', 'Montenegro', 'Sweden', 'Hungary',
    'Namibia', 'Bahrain', 'Netherlands', 'New Zealand', 'China', 'Ivory Coast',
    'Moldova', 'Guadeloupe', 'Czech Republic', 'Albania', 'Sierra Leone',
    'Tunisia', 'Chad', 'Azerbaijan', 'Papua New Guinea', 'Malaysia', 'Guinea',
    'Latvia', 'Suriname', 'Morocco', 'Vietnam', 'Belarus', 'Bosnia-Herzegovina',
    'Uruguay', 'New Caledonia', 'Kuwait', 'Serbia', 'Brunei', 'Macau', 'Zaire',
    'Madagascar', 'Norway', 'Zambia', 'Rhodesia', 'Djibouti', 'Gabon', 'Kazakhstan',
    'Costa Rica', 'French Guiana', 'Poland', 'Uzbekistan', 'Guyana', 'Lithuania',
    'East Timor', 'Bulgaria', 'Gambia', 'Mauritius', 'Czechoslovakia', 'Croatia',
    'Belize', 'Finland', 'Republic of the Congo', 'Ghana', 'North Yemen',
    "People's Republic of the Congo", 'Malta', 'Luxembourg', 'Soviet Union',
    'Mauritania', 'Togo', 'Lesotho', 'Eritrea', 'Hong Kong', 'Swaziland',
    'Trinidad and Tobago', 'Grenada', 'Guinea-Bissau', 'Dominica', 'Malawi',
    'Andorra', 'Estonia', 'Vanuatu', 'Fiji', 'Benin', 'Qatar', 'Turkmenistan',
    'Bhutan', 'French Polynesia', 'Bahamas', 'Comoros', 'Slovenia', 'Singapore',
    'Vatican City', 'Other'
]

country_provinces = {
    "Afghanistan": ["Zabul", "Kabul", "Faryab", "Laghman"],
    "Albania": ["Tirana", "Durres", "Fier", "Vlore"],
    "Algeria": ["Algiers", "Boumerdes", "Tizi Ouzou", "Oran"],
    "Egypt": ["Cairo", "Alexandria", "Giza", "North Sinai"],
    "Iraq": ["Baghdad", "Basra", "Mosul", "Kirkuk"],
    "Andorra": ["Unknown", "Unknown", "Unknown", "Unknown"],
    "Angola": ["Luanda", "Bengo", "Huambo", "Cuanza Norte"],
    "Argentina": ["Buenos Aires", "Cordoba", "Mendoza", "Santa Fe"],
    "Armenia": ["Yerevan", "Shirak", "Aragatsotn", "Kotayk"],
    "Australia": ["New South Wales", "Victoria", "Queensland", "Tasmania"],
    "Austria": ["Vienna", "Tyrol", "Styria", "Salzburg"],
    "Bahrain": ["Central", "Southern", "Northern", "Capital"],
    "Bangladesh": ["Dhaka", "Khulna", "Chittagong", "Rajshahi"],
    "India": ["Bihar", "Delhi", "Karnataka", "Maharashtra"],
    "Pakistan": ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan"],
    "United States": ["New York", "California", "Texas", "Florida"],
    "Other": ["Unknown", "Unknown", "Unknown", "Unknown"]
}


with col1:
    st.markdown("### Temporal and Spatial Information")
    year = st.number_input("Year", min_value=1970, max_value=2030, value=2020, help="Enter the year of the incident.")
    month = st.selectbox("Month", list(range(1, 13)), help="Select the month (1-12).")
    country = st.selectbox("Country", countries)

    # If user selects "Other", allow manual input
    if country == "Other":
        country = st.text_input("Please enter the country name:")
        region = st.selectbox("Region", [
            "Middle East & North Africa", "South Asia", "Europe", 
            "Sub-Saharan Africa", "North America", "East Asia", "Other"
        ])
    else:
        region = "Unknown"

    provinces = country_provinces.get(country, ["Unknown", "Unknown", "Unknown", "Unknown"])
    province_state = st.selectbox("Province / State", provinces)

    if province_state == "Unknown":
        province_state = st.text_input("Please enter the province or state name:")

    st.markdown("### Motivational Criteria")
    criteria_political = st.selectbox("Political Criteria", [0, 1])
    criteria_economic = st.selectbox("Economic Criteria", [0, 1])
    criteria_religious = st.selectbox("Religious Criteria", [0, 1])
    multiple_attacks = st.selectbox("Multiple Attacks", [0, 1, "Unknown"])

# ---------------- Attack Details ----------------
with col2:
    st.markdown("### Attack Characteristics")
    suicide_attack = st.selectbox("Suicide Attack", [0, 1])
    primary_attack_type = st.selectbox("Primary Attack Type", [
        "Bombing/Explosion", "Assassination", "Armed Assault", "Hijacking", 
        "Hostage Taking", "Facility/Infrastructure Attack"
    ])
    primary_target_type = st.selectbox("Primary Target Type", [
        "Unknown","Business","Military","Private Citizens & Property",
        "Violent Political Party","Police","Government (General)",
        "Religious Figures/Institutions","Journalists & Media","Airports & Aircraft",
        "Other","Educational Institution","Government (Diplomatic)","Utilities",
        "Food or Water Supply","Transportation","NGO","Abortion Related",
        "Terrorists/Non-State Militia","Telecommunication","Maritime","Tourists"
    ])
    primary_target_subtype = st.selectbox("Primary Target Subtype", [
        "Unknown","Police Security Forces/Officers","Military Personnel (soldiers, troops, officers, forces)",
        "Unnamed Civilian/Unspecified","Politician or Political Party Movement/Meeting/Rally",
        "Government Personnel (excluding police, military)","Village/City/Town/Suburb",
        "Military Barracks/Base/Headquarters/Checkpost","Military Unit/Patrol/Convoy",
        "Police Building (headquarters, station, school)","Police Patrol (including vehicles and convoys)",
        "Government Building/Facility/Office","Retail/Grocery/Bakery","School/University/Educational Building",
        "Military Checkpoint","Political Party Member/Rally","House/Apartment/Residence",
        "Place of Worship","Marketplace/Plaza/Square","Non-State Militia","Train/Train Tracks/Trolley",
        "Laborer (General)/Occupation Identified","Vehicles/Transportation","Named Civilian"
    ])
    target_nationality = st.text_input("Target Nationality", "Egyptian")
    secondary_target_type = st.selectbox("Secondary Target Type", [
        "Unknown","Private Citizens & Property","Military","Police",
        "Terrorists/Non-State Militia","Business","Government (General)",
        "Educational Institution","Transportation","Religious Figures/Institutions",
        "Government (Diplomatic)","Journalists & Media","NGO","Utilities",
        "Tourists","Violent Political Party","Airports & Aircraft","Telecommunication",
        "Other","Maritime","Food or Water Supply"
    ])

    st.markdown("### Perpetrator Information")
    group_options = [
        "Unknown","Taliban","Islamic State of Iraq and the Levant (ISIL)",
        "Houthi extremists (Ansar Allah)","Al-Shabaab","Other"
    ]
    group_name = st.selectbox("Group Name", group_options)
    if group_name == "Other":
        group_name = st.text_input("Please enter the group name:")

    individual_attack = st.selectbox("Individual Attack", [0, 1])
    primary_weapon_type = st.selectbox("Primary Weapon Type", [
        "Firearms","Explosives","Melee","Incendiary","Chemical","Unknown"
    ])
    secondary_weapon_type = st.selectbox("Secondary Weapon Type", [
        "None","Firearms","Explosives","Melee","Incendiary","Chemical","Unknown"
    ])

# ================================
# Prepare DataFrame
# ================================
input_data = pd.DataFrame({
    "year": [year],
    "month": [month],
    "country": [country],
    "region": [region],
    "province_state": [province_state],
    "criteria_political": [criteria_political],
    "criteria_economic": [criteria_economic],
    "criteria_religious": [criteria_religious],
    "multiple_attacks": [multiple_attacks],
    "suicide_attack": [suicide_attack],
    "primary_attack_type": [primary_attack_type],
    "primary_target_type": [primary_target_type],
    "primary_target_subtype": [primary_target_subtype],
    "target_nationality": [target_nationality],
    "secondary_target_type": [secondary_target_type],
    "group_name": [group_name],
    "individual_attack": [individual_attack],
    "primary_weapon_type": [primary_weapon_type],
    "secondary_weapon_type": [secondary_weapon_type]
})

st.markdown("---")

# ================================
# Prediction Section
# ================================
st.header("Prediction Results")
st.caption("Click **Predict Attack Success** to estimate the probability of success based on your input.")
if st.button("Predict Attack Success"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"**Likely Successful Attack** — Probability: {probability:.2%}")
        else:
            st.success(f"**Likely Unsuccessful Attack** — Probability: {probability:.2%}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")

# ================================
# Footer
# ================================
st.markdown(
    "<p style='text-align:center; font-size:16px; color:#5c636a;'><i>Turning data into defense — where intelligence meets insight</i><br><span style='font-size:14px; color:#8b8b8b;'>© 2025 | <b>Alaa Gabr</b></span></p>",
    unsafe_allow_html=True
)


