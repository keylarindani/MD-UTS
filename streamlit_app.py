import streamlit as st
import pickle
import pandas as pd

class HotelBookingApp:
    def __init__(self):
        self.model = self.load_model('best_model_rf (2).pkl')  # Random Forest after tuning
        self.encoder = self.load_model('encoder (2).pkl')  # OneHotEncoder dari training
        self.data = self.load_csv('Dataset_B_hotel.csv')  # Optional preview

    def load_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_csv(self, path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load CSV file: {e}")
            return None

    def encode_input(self, input_df):
        try:
            encoded = self.encoder.transform(input_df)
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(),
                index=input_df.index
            )
            return encoded_df
        except Exception as e:
            st.error(f"❌ Encoding Error: {e}")
            return None

    def predict(self, input_df):
        encoded_df = self.encode_input(input_df)
        if encoded_df is not None:
            prediction = self.model.predict(encoded_df)[0]
            probability = self.model.predict_proba(encoded_df)[0][1]
            return prediction, probability
        else:
            return None, None

    def run(self):
        st.title("🏨 Hotel Booking Cancellation Prediction")
        st.caption("Model: Random Forest (after tuning)")
        st.markdown("---")

        if self.data is not None:
            with st.expander("📂 Preview Dataset"):
                st.dataframe(self.data.head(20))
            st.markdown("---")

        st.subheader("✏️ Input Booking Data")

        test_cases = {
            "Test Case 1": {
                'lead_time': 45,
                'no_of_adults': 2,
                'no_of_children': 0,
                'no_of_weekend_nights': 1,
                'no_of_week_nights': 2,
                'type_of_meal_plan': 'Meal Plan 1',
                'required_car_parking_space': 0,
                'room_type_reserved': 'Room_Type 1',
                'market_segment_type': 'Online',
                'repeated_guest': 0,
                'no_of_previous_cancellations': 0,
                'no_of_previous_bookings_not_canceled': 0,
                'avg_price_per_room': 100.0,
                'no_of_special_requests': 1
            },
            "Test Case 2": {
                'lead_time': 100,
                'no_of_adults': 1,
                'no_of_children': 2,
                'no_of_weekend_nights': 2,
                'no_of_week_nights': 5,
                'type_of_meal_plan': 'Meal Plan 2',
                'required_car_parking_space': 1,
                'room_type_reserved': 'Room_Type 3',
                'market_segment_type': 'Offline',
                'repeated_guest': 1,
                'no_of_previous_cancellations': 1,
                'no_of_previous_bookings_not_canceled': 3,
                'avg_price_per_room': 150.0,
                'no_of_special_requests': 2
            }
        }

        selected = st.selectbox("Pilih Input:", ["Manual Input"] + list(test_cases.keys()))

        if selected != "Manual Input":
            input_df = pd.DataFrame([test_cases[selected]])
        else:
            input_df = pd.DataFrame([{
                'lead_time': st.slider('Lead Time', 0, 500, 45),
                'no_of_adults': st.slider('Jumlah Dewasa', 1, 5, 2),
                'no_of_children': st.slider('Jumlah Anak-anak', 0, 5, 0),
                'no_of_weekend_nights': st.slider('Weekend Nights', 0, 5, 1),
                'no_of_week_nights': st.slider('Week Nights', 0, 10, 2),
                'type_of_meal_plan': st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
                'required_car_parking_space': st.selectbox('Car Parking Needed?', [0, 1]),
                'room_type_reserved': st.selectbox('Room Type', [
                    'Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'
                ]),
                'market_segment_type': st.selectbox('Market Segment', ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary']),
                'repeated_guest': st.selectbox('Repeated Guest?', [0, 1]),
                'no_of_previous_cancellations': st.slider('Previous Cancellations', 0, 5, 0),
                'no_of_previous_bookings_not_canceled': st.slider('Previous Non-Cancelled Bookings', 0, 10, 0),
                'avg_price_per_room': st.number_input('Average Price per Room', min_value=0.0, value=100.0),
                'no_of_special_requests': st.slider('Special Requests', 0, 5, 0)
            }])

        if st.button("🔮 Predict Booking Status"):
            pred, prob = self.predict(input_df)
            if pred is not None:
                label = "✅ Not Cancelled" if pred == 0 else "❌ Cancelled"
                st.success(f"### Prediksi: {label}")
                st.info(f"Probabilitas Pembatalan: {prob:.2%}")
                st.markdown("#### 🔎 Input Data")
                st.dataframe(input_df)
            else:
                st.error("Gagal melakukan prediksi. Pastikan semua input valid.")

if __name__ == '__main__':
    app = HotelBookingApp()
    app.run()
