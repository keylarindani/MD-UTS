import streamlit as st
import pandas as pd
import pickle

# ========== Load Model dan Encoder ==========
with open("best_model_rf.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

st.title("🔍 Prediksi Pembatalan Booking Hotel")

# ========== Form Input ==========
st.subheader("📥 Silakan isi data pemesanan:")

col1, col2 = st.columns(2)
with col1:
    lead_time = st.number_input("Lead Time (hari)", min_value=0)
    no_of_adults = st.number_input("Jumlah Dewasa", min_value=0)
    no_of_children = st.number_input("Jumlah Anak", min_value=0)
    no_of_weekend_nights = st.number_input("Malam Akhir Pekan", min_value=0)
    no_of_week_nights = st.number_input("Malam Hari Kerja", min_value=0)
    avg_price = st.number_input("Harga Rata-Rata Kamar", min_value=0.0)

with col2:
    meal_plan = st.selectbox("Paket Makanan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    room_type = st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    market_segment = st.selectbox("Segment Pasar", ["Offline", "Online", "Corporate", "Aviation", "Complementary"])
    parking = st.selectbox("Butuh Parkir?", [0, 1])
    repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
    special_request = st.number_input("Jumlah Permintaan Khusus", min_value=0)

cancel_before = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0)
not_cancel_before = st.number_input("Jumlah Booking Sukses Sebelumnya", min_value=0)

# ========== Tombol Test Case ==========
if st.button("🧪 Load Test Case 1"):
    lead_time = 50
    no_of_adults = 2
    no_of_children = 1
    no_of_weekend_nights = 2
    no_of_week_nights = 3
    avg_price = 90.0
    meal_plan = "Meal Plan 1"
    room_type = "Room_Type 1"
    market_segment = "Online"
    parking = 1
    repeated_guest = 0
    special_request = 1
    cancel_before = 0
    not_cancel_before = 3
    st.warning("✅ Test Case 1 dimuat. Silakan klik tombol prediksi.")

if st.button("🧪 Load Test Case 2"):
    lead_time = 250
    no_of_adults = 1
    no_of_children = 0
    no_of_weekend_nights = 0
    no_of_week_nights = 1
    avg_price = 200.0
    meal_plan = "Not Selected"
    room_type = "Room_Type 3"
    market_segment = "Offline"
    parking = 0
    repeated_guest = 0
    special_request = 0
    cancel_before = 2
    not_cancel_before = 0
    st.warning("❌ Test Case 2 dimuat. Silakan klik tombol prediksi.")

# ========== Tombol Prediksi ==========
if st.button("🔮 Prediksi"):
    # Susun data user jadi DataFrame
    input_df = pd.DataFrame([{
        'lead_time': lead_time,
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': meal_plan,
        'required_car_parking_space': parking,
        'room_type_reserved': room_type,
        'market_segment_type': market_segment,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': cancel_before,
        'no_of_previous_bookings_not_canceled': not_cancel_before,
        'avg_price_per_room': avg_price,
        'no_of_special_requests': special_request
    }])

    # Encode kolom kategorikal
    cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    encoded = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    final_input = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)

    # Prediksi
    prediction = model.predict(final_input)[0]
    result = "❌ Booking DIBATALKAN" if prediction == 1 else "✅ Booking TIDAK dibatalkan"

    # Output
    st.subheader("📊 Hasil Prediksi:")
    st.success(result)
