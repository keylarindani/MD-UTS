import streamlit as st
import pandas as pd
import pickle

# ===== Load Model dan Encoder =====
with open("best_model_rf (2).pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder (2).pkl", "rb") as f:
    encoder = pickle.load(f)

with open("expected_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)  # Daftar kolom saat training

st.title("Prediksi Pembatalan Booking Hotel")

# ===== Ambil input dari user =====
lead_time = st.number_input("Lead Time", min_value=0)
no_of_adults = st.number_input("Jumlah Dewasa", min_value=0)
no_of_children = st.number_input("Jumlah Anak", min_value=0)
no_of_weekend_nights = st.number_input("Malam Akhir Pekan", min_value=0)
no_of_week_nights = st.number_input("Malam Hari Kerja", min_value=0)

meal = st.selectbox("Paket Makanan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
room = st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
market = st.selectbox("Segmen Pasar", ["Offline", "Online", "Corporate", "Aviation", "Complementary"])

parking = st.selectbox("Butuh Parkir?", [0, 1])
repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
cancelled_before = st.number_input("Pembatalan Sebelumnya", min_value=0)
not_cancelled_before = st.number_input("Booking Sukses Sebelumnya", min_value=0)
avg_price = st.number_input("Harga Rata-Rata Kamar", min_value=0.0)
special_request = st.number_input("Jumlah Permintaan Khusus", min_value=0)

# ===== Buat DataFrame =====
input_df = pd.DataFrame([{
    'lead_time': lead_time,
    'no_of_adults': no_of_adults,
    'no_of_children': no_of_children,
    'no_of_weekend_nights': no_of_weekend_nights,
    'no_of_week_nights': no_of_week_nights,
    'type_of_meal_plan': meal,
    'room_type_reserved': room,
    'market_segment_type': market,
    'required_car_parking_space': parking,
    'repeated_guest': repeated_guest,
    'no_of_previous_cancellations': cancelled_before,
    'no_of_previous_bookings_not_canceled': not_cancelled_before,
    'avg_price_per_room': avg_price,
    'no_of_special_requests': special_request
}])

# ===== Pisah kolom kategorikal dan numerik =====
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
input_encoded = encoder.transform(input_df[categorical_cols])
input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))

input_numeric_df = input_df.drop(columns=categorical_cols)
final_input = pd.concat([input_numeric_df, input_encoded_df], axis=1)

# ===== Pastikan kolom tersusun sesuai expected_columns =====
final_input = final_input.reindex(columns=expected_columns, fill_value=0)

# ===== Prediksi =====
if st.button("🔮 Prediksi"):
    pred = model.predict(final_input)[0]
    hasil = "Booking Dibatalkan ❌" if pred == 1 else "Booking Tidak Dibatalkan ✅"
    st.success(f"Hasil Prediksi: **{hasil}**")
