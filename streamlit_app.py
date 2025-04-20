import streamlit as st
import pickle
import numpy as np
import pandas as pd

class HotelBookingApp:
    def __init__(self):
        self.model = self.load_model('best_model_rf.pkl')  # atau xgboost_model.pkl
        self.encoder = self.load_model('encoder.pkl')
        self.data = self.load_csv('Dataset_B_hotel.csv')

    def load_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_csv(self, path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat CSV: {e}")
            return None

    def encode_input(self, input_df):
        cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        encoded_array = self.encoder.transform(input_df[cat_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out(cat_cols), index=input_df.index)
        df_numeric = input_df.drop(columns=cat_cols)
        final_input = pd.concat([df_numeric, encoded_df], axis=1)
        return final_input

    def predict(self, input_df):
        final_input = self.encode_input(input_df)
        prediction = self.model.predict(final_input)[0]
        probability = self.model.predict_proba(final_input)[0][1]
        return prediction, probability

    def run(self):
        st.title("🏨 Prediksi Pembatalan Booking Hotel")
        st.write("Perkirakan apakah pemesanan akan dibatalkan atau tidak.")

        if self.data is not None:
            st.subheader("📄 Preview Dataset")
            st.dataframe(self.data.head())

        st.subheader("📝 Input Booking")

        test_cases = {
            "Test Case 1": {
                'lead_time': 45, 'no_of_adults': 2, 'no_of_children': 0,
                'no_of_weekend_nights': 1, 'no_of_week_nights': 2,
                'type_of_meal_plan': 'Meal Plan 1', 'room_type_reserved': 'Room_Type 1',
                'market_segment_type': 'Online', 'required_car_parking_space': 0,
                'repeated_guest': 0, 'no_of_previous_cancellations': 0,
                'no_of_previous_bookings_not_canceled': 0,
                'avg_price_per_room': 100.0, 'no_of_special_requests': 1
            },
            "Test Case 2": {
                'lead_time': 100, 'no_of_adults': 1, 'no_of_children': 2,
                'no_of_weekend_nights': 2, 'no_of_week_nights': 5,
                'type_of_meal_plan': 'Meal Plan 2', 'room_type_reserved': 'Room_Type 3',
                'market_segment_type': 'Offline', 'required_car_parking_space': 1,
                'repeated_guest': 1, 'no_of_previous_cancellations': 1,
                'no_of_previous_bookings_not_canceled': 3,
                'avg_price_per_room': 150.0, 'no_of_special_requests': 2
            }
        }

        selected = st.selectbox("🧪 Pilih Test Case", ["Manual Input"] + list(test_cases.keys()))

        if selected != "Manual Input":
            user_input = pd.DataFrame([test_cases[selected]])
        else:
            user_input = pd.DataFrame([{
                'lead_time': st.slider('Lead Time (hari)', 0, 500, 45),
                'no_of_adults': st.number_input('Jumlah Dewasa', 1, 10, 2),
                'no_of_children': st.number_input('Jumlah Anak', 0, 10, 0),
                'no_of_weekend_nights': st.number_input('Malam Weekend', 0, 10, 1),
                'no_of_week_nights': st.number_input('Malam Weekday', 0, 10, 2),
                'type_of_meal_plan': st.selectbox('Paket Makanan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
                'room_type_reserved': st.selectbox('Tipe Kamar', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']),
                'market_segment_type': st.selectbox('Segment Pasar', ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary']),
                'required_car_parking_space': st.selectbox('Butuh Parkir?', [0, 1]),
                'repeated_guest': st.selectbox('Tamu Berulang?', [0, 1]),
                'no_of_previous_cancellations': st.slider('Pembatalan Sebelumnya', 0, 10, 0),
                'no_of_previous_bookings_not_canceled': st.slider('Booking Sukses Sebelumnya', 0, 10, 0),
                'avg_price_per_room': st.number_input('Harga Kamar', 0.0, 1000.0, 100.0),
                'no_of_special_requests': st.slider('Permintaan Khusus', 0, 5, 1)
            }])

        if st.button("🔮 Prediksi"):
            pred, prob = self.predict(user_input)
            hasil = "✅ Tidak Dibatalkan" if pred == 0 else "❌ Dibatalkan"
            st.success(f"### Hasil Prediksi: {hasil}")
            st.info(f"### Probabilitas: {prob:.2%}")
            st.dataframe(user_input)

if __name__ == "__main__":
    app = HotelBookingApp()
    app.run()
