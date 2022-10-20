# Predictive Analysis of Hotel Booking Cancellations
by [Cindy Fransisca](https://www.linkedin.com/in/cindy-fransisca-18ba81213/)

##  Business Understanding

### Introduction

Booking cancellations have a significant impact in the hospitality industry. Some hotels use the overbooking strategy to avoid booking cancellation problems. The overbooking strategy can help maximizing the revenue but might also give a negative impact on reputation if the hotel is actually overbooked. Predicting the hotel booking cancellations can help the hotel businesses improve their overbooking strategy and minimize their risk.

**Target** :
- 1 : Cancelled Booking
- 0 : Non-Cancelled Booking

### Problem Statement :

The overbooking strategy will perform well if we know the number of bookings that will be cancelled by the customers. If the assumed number of cancelled bookings is too small, then we will be left with a lot of empty rooms. If the assumed number of cancelled bookings is too big, then we will have a lot of upset customers who don't get rooms. 


### Goals :

Based on the problems we have, we want to detect which bookings will be cancelled or not. This way we can make better decisions on the overbooking strategy. We might also find which characteristics resulting in cancelled bookings, so it will help us identify the cancelled bookings better.


### Analytic Approach :

We will analyze the data to find the characteristics of cancelled bookings. After that, we will build a classification model to help predicting the number of cancelled bookings.

### Evaluation Metric

Type 1 error : False Positive  
cons: Rooms will be overbooked, and customers will be disappointed (no longer book the hotel)

Type 2 error : False Negative  
cons: Rooms are empty, even though there might be potential customers who want to stay at the hotel

Both cases will have negative impacts on the revenue, but the overbooked cases are worse because they can cause potential loss of future revenue from discontent customers who will no longer book the hotel. In this case our positive target is more important than the negative class. We might also put more weight on precision to avoid a lot of upset customers. So, the main metric that will be used is f0.5-score.


## Data Understanding

Dataset source : https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

Note : 
- Target is imbalance in the dataset
- Each row represents informations of a booking which is made by past customers.

### Attribute Information

| Attribute | Description |
| --- | --- |
| country | Country of origin. Categories are represented in the ISO 3155–3:2013 format. |
| market_segment | Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”. |
| previous_cancellations | Number of previous bookings that were cancelled by the customer prior to the current booking. |
| booking_changes | Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation. |
| deposit_type | Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay. |
| days_in_waiting_list | Number of days the booking was in the waiting list before it was confirmed to the customer. |
| customer_type | Type of booking, assuming one of four categories: Contract - when the booking has an allotment or other type of contract associated to it; Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking. |
| reserved_room_type | Code of room type reserved. Code is presented instead of designation for anonymity reasons. |
| required_car_parking_spaces | Number of car parking spaces required by the customer. |
| total_of_special_requests | Number of special requests made by the customer (e.g. twin bed or high floor). |
| is_canceled | Value indicating if the booking was canceled (1) or not (0). |


## Preprocessing

We will do several steps for our preprocessing:
- Impute missing values
    - `country`: constant (Other)
        <br>
        We will fill the missing values in the `country` column with 'Other' because we can't guess where the bookings come from. So, it will make more sense to fill them with 'Other'.
- Encoding
    - `country`: binary encoding
       <br>
        We will use binary encoding for the `country` column because it has a large number of uniques
    - `market_segment`, `deposit_type`, `customer_type`, `reserved_room_type` : one-hot encoding
      <br>
        For the other category columns, we will use one-hot encoding because the data is nominal not ordinal and they don't have large number of uniques.  
- Scaling : Robust Scaler
   <br>
    We use Robust Scaler because our data has outliers and Robust Scaler won't be affected by outliers. We can tune the scaler later to find the best scaler.

## Model Implementation

Let's say there are 200 bookings for a hotel (100 canceled and 100 not canceled) and the hotel has 200 rooms. A room that is filled with customer can generate 125 USD for the hotel revenue. If a room is empty, it will cost the hotel 25 USD and if a room is overbooked, it will cost the hotel 100 USD for customer compensation. For this example, we will assume that the hotel can find substitute customers for every predicted canceled rooms.

---
**Without the model, we will assume all bookings are canceled.**

- predicted number of canceled bookings = 200
- actual number of canceled bookings = 100

Because the actual number of canceled bookings is smaller than the predicted one, it means we have overbooked rooms in our hotel. The number of overbooked rooms equal to the difference between the actual and predicted number of canceled bookings.

- filled rooms = 200 x 125 USD = 25.000 USD
- empty rooms = 0 x -25 USD = 0 USD
- overbooked rooms = 100 x -100 USD = -10.000 USD

net revenue = 25.000 - 0 - 10.000 = **15.000 USD**

(This calculation hasn't counted the cost for potential loss of future revenue and bad reputation)

---

**With our model, we will overbook the rooms based on the model canceled predictions**

- predicted number of canceled bookings = 81
- actual number of canceled bookings = 100

Because the actual number of canceled bookings is bigger than the predicted one, it means we have empty rooms in our hotel. The number of empty rooms equal to the difference between the actual and predicted number of canceled bookings.

- filled rooms = 181 x 125 USD = 22.625 USD
- empty rooms = 19 x -25 USD = -475 USD
- overbooked rooms = 0 x -50 USD = 0 USD

net revenue = 12.500 - 475 - 0 = **22.150 USD**

(There are no upset customers, which is good for the hotel reputation)

---

**Increase in revenue = 22.150 - 15.000 = 7.150 USD**

% Increase = 7.150 / 15.000 x 100 = 47,67%

Based on the example, our model can increase **47,67%** of the hotel's net revenue while still maintaining good reputation for the hotel (no overbook rooms).


## Model limitation

There are some limitations in our model
- **Lack of data**<br>
    There are 2 type of limitation: lack of data and lack of **good** data. There is a saying that goes 'AI can only be as smart as the quality of data'. In this case, our model can only predict accurately if the prediction data is within the range of the training data. For example, most of the country in the training data come from Europe, so our model might be biased for country outside Europe.
- **Lack of interpretability** <br>
    A complex ensemble model usually perform better in discovering underlying patterns and increasing accuracy. Even though our model can predict accurate results, it is hard for us to explain how our model arrived at the conclusion. 
- **Computational and Time limitation** <br>
    Hyperparameter tuning is a crucial part for improving our model performance, but some models require sufficient amount of computing power and time. For example, we only try 50 iterations in randomized search for xgboost, so our model hasn't improved much. 
    
    
## Conclusion

From the results, we can summarize our final model using Xgboost:
- Top 3 most important features are **deposit_type_Non Refund**, **required_car_parking_spaces, and market_segment_Online TA**
- **Precision** = 0.77  which mean the chance of our model can predict a canceled booking correctly are **77%**.
- **Recall** =  0.69 which mean out of 100% of actually canceled bookings, our model can predict **69%** of them.
- **False Positive Rate** (actually not canceled, predicted canceled) = **12%**
- **False Negative Rate** (actually canceled, predicted not canceled) = **31%**
- Implementation of model can **increase  47,67% of the hotel's net revenue**
- Model limitation = **lack of data, lack of interpretability, computational and time limitation**


## Recommendation

Some recommendations to improve our model:
- Collecting more good data to make our model more robust. A lot of country has less than 10 data, so the data is highly imbalanced.
- Adding new features that might be related to canceled bookings, for example: the the date of arrival (to analyze the trend), number of days between booking and arrival, Average Daily Rate (ADR), etc.
- If we have sufficient high-quality data, we might try to use a simple single-type model to increase the interpretability of our model. Usually, a simple model can do the job if the data quality is good.
- Try different combination of feature selection and feature engineering, for example changing the base model for selecting feature importance or use SHAP.
