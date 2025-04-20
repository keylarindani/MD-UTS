import streamlit as st
import pickle
import numpy as np
import pandas as pd

class HotelBookingApp:
    def __init__(self):
        self.model = self.load_pickle('best_model_rf (2).pkl')  # <- best model after tuning
        self.encoder = self.load_pickle('encoder (2).pkl')  # assumed OneHotEncoder
        self.data = self.load_csv('Dataset_B_hotel.csv')  # optional: preview raw data

    def load_pickle(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Gagal memuat file {path}: {e}")
            return None

    def load_csv(self, path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"⚠️ Gagal load dataset: {e}")
            return None

    def encode_input(self, input_df):
        try:
            # Transform only columns seen during training
            encoded = self.encoder.transform(input_df)
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(), index=input_df.index)
            return encoded_df
        except Exception as e:
            st.error(f"❌ Encoding gagal: {e}")
            return None

    def predict(self, input_df):
        encoded_df = self.encode_input(input_df)
        if encoded_df is not None:
            prediction = self.model.predict(encoded_df)[0]
            probability = self.model.predict_proba(encoded_df)[0][1]  # prob class 1 (Cancelled)
            return prediction, probability
        return None, None

    def run(self):
        st.title("🧠 Hotel Booking Cancellation Prediction App")
        st.caption("Model: Random Forest (After Tuning)")
        st.markdown("---")

        if self.data is not None:
            with st.expander("📊 Preview Dataset"):
                st.dataframe(self.data.head())

        st.subheader("📝 Input Data Booking")

        # Input fields
        input_dict = {
            'lead_time': st.slider('Lead Time', 0, 500, 50),
            'no_of_adults': st.slider('Jumlah Dewasa', 1, 5, 2),
            'no_of_children': st.slider('Jumlah Anak-anak', 0, 5, 0),
            'no_of_weekend_nights': st.slider('Malam Akhir Pekan', 0, 5, 1),
            'no_of_week_nights': st.slider('Malam Hari Kerja', 0, 10, 2),
            'type_of_meal_plan': st.selectbox('Paket Makanan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
            'required_car_parking_space': st.selectbox('Butuh Parkir?', [0, 1]),
            'room_type_reserved': st.selectbox('Tipe Kamar', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']),
            'market_segment_type': st.selectbox('Segment Pasar', ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary']),
            'repeated_guest': st.selectbox('Tamu Berulang?', [0, 1]),
            'no_of_previous_cancellations': st.slider('Pembatalan Sebelumnya', 0, 10, 0),
            'no_of_previous_bookings_not_canceled': st.slider('Booking Sukses Sebelumnya', 0, 10, 0),
            'avg_price_per_room': st.number_input('Harga Rata-Rata Kamar', min_value=0.0, value=100.0),
            'no_of_special_requests': st.slider('Jumlah Permintaan Khusus', 0, 5, 0),
        }

        input_df = pd.DataFrame([input_dict])

        if st.button("🔮 Prediksi"):
            prediction, probability = self.predict(input_df)
            if prediction is not None:
                label = "❌ Cancelled" if prediction == 1 else "✅ Not Cancelled"
                st.success(f"### Hasil Prediksi: {label}")
                st.info(f"Probabilitas Pembatalan: {probability:.2%}")
                st.markdown("#### 🔎 Data yang Digunakan")
                st.dataframe(input_df)
            else:
                st.warning("Gagal melakukan prediksi.")

if __name__ == '__main__':
    app = HotelBookingApp()
    app.run()
