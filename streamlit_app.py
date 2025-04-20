import streamlit as st
import pickle
import pandas as pd

class HotelBookingApp:
    def __init__(self):
        self.model = self.load_pickle('best_model_rf (4).pkl')
        self.encoder = self.load_pickle('encoder (4).pkl')
        self.data = self.load_csv('Dataset_B_hotel.csv')

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
        input_encoded = input_df.copy()

        # Apply OneHotEncoder (categorical features)
        ohe = self.encoder['onehot']
        ohe_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        ohe_encoded = ohe.transform(input_df[ohe_cols])
        ohe_feature_names = ohe.get_feature_names_out(ohe_cols)
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=input_df.index)

        # Apply LabelEncoder (label encoding target-like binary vars)
        input_encoded.drop(columns=ohe_cols, inplace=True)
        le = self.encoder['binary']
        for col in ['required_car_parking_space', 'repeated_guest']:
            input_encoded[col] = le[col].transform(input_df[col])

        # Final encoded dataframe
        final_encoded = pd.concat([input_encoded, ohe_df], axis=1)
        return final_encoded

    def predict(self, input_df):
        try:
            encoded_input = self.encode_input(input_df)
            prediction = self.model.predict(encoded_input)[0]
            probability = self.model.predict_proba(encoded_input)[0][1]
            return prediction, probability
        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")
            return None, None

    def run(self):
        st.title("🏨 Hotel Booking Cancellation Prediction")
        st.write("Prediksi apakah pemesanan hotel akan **dibatalkan** atau **tidak dibatalkan** berdasarkan data booking.")

        # Show dataset preview
        if self.data is not None:
            st.subheader("📂 Dataset Preview")
            st.dataframe(self.data.head())

        st.markdown("---")
        st.subheader("📝 Input Booking Data")

        # Test Cases
        test_cases = {
            "Test Case 1": {
                'no_of_adults': 2,
                'no_of_children': 0,
                'no_of_weekend_nights': 1,
                'no_of_week_nights': 2,
                'type_of_meal_plan': 'Meal Plan 1',
                'required_car_parking_space': '0',
                'room_type_reserved': 'Room_Type 1',
                'lead_time': 45,
                'arrival_year': 2017,
                'arrival_month': 7,
                'arrival_date': 15,
                'market_segment_type': 'Online',
                'repeated_guest': '0',
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
                'required_car_parking_space': '1',
                'room_type_reserved': 'Room_Type 3',
                'lead_time': 100,
                'arrival_year': 2017,
                'arrival_month': 12,
                'arrival_date': 25,
                'market_segment_type': 'Offline',
                'repeated_guest': '1',
                'no_of_previous_cancellations': 1,
                'no_of_previous_bookings_not_canceled': 3,
                'avg_price_per_room': 150.0,
                'no_of_special_requests': 2
            }
        }

        selected = st.selectbox("📁 Pilih Input:", ["Manual Input"] + list(test_cases.keys()))

        if selected != "Manual Input":
            input_data = pd.DataFrame([test_cases[selected]])
        else:
            input_data = pd.DataFrame([{
                'no_of_adults': st.number_input('👨‍👩‍👧‍👦 Jumlah Dewasa', 1, 10, 2),
                'no_of_children': st.number_input('👶 Jumlah Anak', 0, 10, 0),
                'no_of_weekend_nights': st.number_input('🛏️ Malam Akhir Pekan', 0, 10, 1),
                'no_of_week_nights': st.number_input('🛌 Malam Hari Kerja', 0, 10, 2),
                'type_of_meal_plan': st.selectbox('🍽️ Paket Makanan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
                'required_car_parking_space': st.selectbox('🚗 Butuh Parkir?', ['0', '1']),
                'room_type_reserved': st.selectbox('🛏️ Tipe Kamar', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']),
                'lead_time': st.slider('⏳ Lead Time (hari)', 0, 500, 45),
                'arrival_year': st.selectbox('📅 Tahun Tiba', [2017, 2018]),
                'arrival_month': st.slider('📅 Bulan Tiba', 1, 12, 7),
                'arrival_date': st.slider('📅 Tanggal Tiba', 1, 31, 15),
                'market_segment_type': st.selectbox('📊 Segment Pasar', ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary']),
                'repeated_guest': st.selectbox('👥 Tamu Berulang?', ['0', '1']),
                'no_of_previous_cancellations': st.slider('❌ Pembatalan Sebelumnya', 0, 10, 0),
                'no_of_previous_bookings_not_canceled': st.slider('✅ Booking Sukses Sebelumnya', 0, 10, 0),
                'avg_price_per_room': st.number_input('💰 Harga Rata-Rata Kamar', 0.0, 1000.0, 100.0),
                'no_of_special_requests': st.slider('🌟 Permintaan Khusus', 0, 5, 1)
            }])

        if st.button("🔮 Predict Booking Status"):
            pred, prob = self.predict(input_data)
            if pred is not None:
                status = "✅ Not Cancelled" if pred == 0 else "❌ Cancelled"
                st.success(f"### Prediction: {status}")
                st.info(f"### Cancellation Probability: {prob:.2%}")
                st.markdown("#### 🔍 Data Used for Prediction")
                st.dataframe(input_data)

if __name__ == "__main__":
    app = HotelBookingApp()
    app.run()
