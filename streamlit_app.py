import streamlit as st
import pickle
import numpy as np
import pandas as pd

class HotelBookingApp:
    def __init__(self):
        self.model = self.load_pickle('best_model_rf (2).pkl')  # Ganti nama file jika beda
        self.encoders = self.load_pickle('encoder (2).pkl')          # berisi {'ohe': ..., 'binary': ...}
        self.data = self.load_csv('Dataset_B_hotel.csv')         # untuk preview saja

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_csv(self, path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load CSV file: {e}")
            return None

    def encode_input(self, input_df):
        df = input_df.copy()

        categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        binary_cols = ['repeated_guest']
        numerical_cols = [col for col in df.columns if col not in categorical_cols + binary_cols]

        # Encode categorical with OneHot
        ohe = self.encoders['ohe']
        ohe_encoded = ohe.transform(df[categorical_cols])
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)

        # Encode binary with LabelEncoder
        le = self.encoders['binary']
        df[binary_cols] = df[binary_cols].apply(lambda col: le.transform(col))

        final_df = pd.concat([df[numerical_cols], df[binary_cols], ohe_df], axis=1)
        return final_df

    def predict(self, input_df):
        encoded_df = self.encode_input(input_df)
        prediction = self.model.predict(encoded_df)[0]
        probability = self.model.predict_proba(encoded_df)[0][1]
        return prediction, probability

    def run(self):
        st.title("🏨 Hotel Booking Cancellation Prediction")
        st.write("Prediksi apakah pemesanan hotel akan **dibatalkan** atau **tidak dibatalkan** berdasarkan informasi yang diberikan.")
        st.markdown("---")

        if self.data is not None:
            st.subheader("📂 Dataset Preview")
            st.dataframe(self.data.head(50))
            st.markdown("---")

        st.subheader("✏️ Input Booking Data")

        test_cases = {
            "Test Case 1": {
                'no_of_adults': 2,
                'no_of_children': 0,
                'no_of_weekend_nights': 1,
                'no_of_week_nights': 2,
                'type_of_meal_plan': 'Meal Plan 1',
                'required_car_parking_space': 0,
                'room_type_reserved': 'Room_Type 1',
                'lead_time': 45,
                'arrival_year': 2017,
                'arrival_month': 7,
                'arrival_date': 15,
                'market_segment_type': 'Online',
                'repeated_guest': 0,
                'no_of_previous_cancellations': 0,
                'no_of_previous_bookings_not_canceled': 0,
                'avg_price_per_room': 100.0,
                'no_of_special_requests': 1
            },
            "Test Case 2": {
                'no_of_adults': 1,
                'no_of_children': 2,
                'no_of_weekend_nights': 2,
                'no_of_week_nights': 5,
                'type_of_meal_plan': 'Meal Plan 2',
                'required_car_parking_space': 1,
                'room_type_reserved': 'Room_Type 3',
                'lead_time': 100,
                'arrival_year': 2017,
                'arrival_month': 12,
                'arrival_date': 25,
                'market_segment_type': 'Offline',
                'repeated_guest': 1,
                'no_of_previous_cancellations': 1,
                'no_of_previous_bookings_not_canceled': 3,
                'avg_price_per_room': 150.0,
                'no_of_special_requests': 2
            }
        }

        selected_case = st.selectbox("📁 Pilih Input:", ["Manual Input"] + list(test_cases.keys()))

        if selected_case != "Manual Input":
            user_input = pd.DataFrame([test_cases[selected_case]])
        else:
            user_input = pd.DataFrame([{
                'no_of_adults': st.number_input('👥 Jumlah Dewasa', 1, 10, 2),
                'no_of_children': st.number_input('🧒 Jumlah Anak', 0, 10, 0),
                'no_of_weekend_nights': st.slider('🛌 Malam Akhir Pekan', 0, 10, 1),
                'no_of_week_nights': st.slider('📅 Malam Hari Kerja', 0, 10, 2),
                'type_of_meal_plan': st.selectbox('🍽 Paket Makanan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
                'required_car_parking_space': st.selectbox('🚗 Parkir Dibutuhkan?', [0, 1]),
                'room_type_reserved': st.selectbox('🏨 Tipe Kamar', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']),
                'lead_time': st.slider('⏳ Lead Time (hari)', 0, 500, 45),
                'arrival_year': st.selectbox('📆 Tahun Kedatangan', [2017, 2018]),
                'arrival_month': st.slider('📆 Bulan Kedatangan', 1, 12, 7),
                'arrival_date': st.slider('📆 Tanggal Kedatangan', 1, 31, 15),
                'market_segment_type': st.selectbox('📊 Segment Pasar', ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary']),
                'repeated_guest': st.selectbox('🔁 Tamu Berulang?', [0, 1]),
                'no_of_previous_cancellations': st.slider('❌ Pembatalan Sebelumnya', 0, 10, 0),
                'no_of_previous_bookings_not_canceled': st.slider('✅ Booking Sukses Sebelumnya', 0, 10, 0),
                'avg_price_per_room': st.number_input('💰 Harga Rata-Rata Kamar', 0.0, 1000.0, 100.0),
                'no_of_special_requests': st.slider('⭐ Permintaan Khusus', 0, 5, 1)
            }])

        if st.button("🔮 Predict Booking Status"):
            try:
                pred, prob = self.predict(user_input)
                status = "✅ Not Cancelled" if pred == 0 else "❌ Cancelled"
                st.success(f"### Hasil Prediksi: {status}")
                st.info(f"### Peluang Pembatalan: {prob:.2%}")
                st.markdown("#### 🔎 Data Input:")
                st.dataframe(user_input)
            except Exception as e:
                st.error(f"❌ Error saat prediksi: {e}")

if __name__ == "__main__":
    app = HotelBookingApp()
    app.run()
