from flask import Flask, jsonify, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
from tensorflow import keras

app = Flask(__name__)
cors = CORS(app)

model2 = keras.models.load_model('ML-project\model\nn_model.h5')

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def main():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":
        form_data = request.form

        if form_data['neighborhood_choice'] in {'North Park', 'Lake View', 'Forest Glen', 'Lincoln Park', 'Edgewater', 'Loop', 'North Center', 'Archer Heights', 'Beverly', 'Near North Side', 'West Town', 'Lincoln Square', 'Edison Park', 'New City', 'Gage Park', 'Hyde Park', 'Logan Square', 'Jefferson Park', 'Montclare', 'Near South Side', 'Uptown', 'Morgan Park', 'Mount Greenwood'}:
            review_scores_location = 99

        elif form_data['neighborhood_choice'] in {'Albany Park', 'Dunning', 'Irving Park', 'Rogers Park', 'Avondale', 'Near West Side', 'Portage Park', 'Bridgeport', 'Pullman', 'Garfield Ridge', 'Douglas', 'Norwood Park', 'Lower West Side', 'West Ridge', 'Kenwood', 'Oakland'}:
            review_scores_location = 4.8

        elif form_data['neighborhood_choice'] in {'Armour Square', 'Grand Boulevard', 'Belmont Cragin', 'Ohare', 'Chatham', 'Mckinley Park'}:
            review_scores_location = 4.7

        elif form_data['neighborhood_choice'] in {'Woodlawn', 'Hermosa', 'Greater Grand Crossing', 'Hegewisch', 'Humboldt Park', 'Roseland', 'Austin'}:
            review_scores_location = 4.6

        elif form_data['neighborhood_choice'] in {'South Lawndale', 'West Pullman', 'Ashburn', 'Washington Park', 'Brighton Park'}:
            review_scores_location = 4.5

        elif form_data['neighborhood_choice'] in {'Southshore', 'Englewood'}:
            review_scores_location = 4.4

        elif form_data['neighborhood_choice'] in {'North Lawndale', 'Calumet Heights', 'East Garfield Park'}:
            review_scores_location = 4.3

        elif form_data['neighborhood_choice'] in {'South Deering'}:
            review_scores_location = 4.2

        elif form_data['neighborhood_choice'] in {'South Chicago', 'Chicago Lawn', 'West Garfield Park'}:
            review_scores_location = 4.0

        else:
            review_scores_location = 0.0

        host_response_time = 4.0
        host_response_rate = 100.0
        host_is_superhost = 0.0
        host_listings_count = 0.0
        host_identity_verified = 1.0
        accommodates = form_data['accommodations_choice']
        bathrooms = form_data['bathrooms_choice']
        bedrooms = form_data['bedrooms_choice']
        beds = form_data['beds_choice']
        price = form_data['price_choice']
        minimum_nights = 2.0
        availability_90 = 45.0
        availability_365 = 180.0
        number_of_reviews = 75.0
        review_scores_rating = review_scores_location
        review_scores_accuracy = 4.9
        review_scores_cleanliness = 4.9
        review_scores_checkin = 4.9
        review_scores_communication = 4.9
        review_scores_value = 4.9
        instant_bookable = 0.0

        # Make DataFrame for model
        input_variables = pd.DataFrame([[host_response_time,
                                         host_response_rate,
                                         host_is_superhost,
                                         host_listings_count,
                                         host_identity_verified,
                                         accommodates,
                                         bathrooms,
                                         bedrooms,
                                         beds,
                                         price,
                                         minimum_nights,
                                         availability_90,
                                         availability_365,
                                         number_of_reviews,
                                         review_scores_rating,
                                         review_scores_accuracy,
                                         review_scores_cleanliness,
                                         review_scores_checkin,
                                         review_scores_communication,
                                         review_scores_location,
                                         review_scores_value,
                                         instant_bookable]],
                                       columns=['host_response_time',
                                                'host_response_rate',
                                                'host_is_superhost',
                                                'host_listings_count',
                                                'host_identity_verified',
                                                'accommodates',
                                                'bathrooms',
                                                'bedrooms',
                                                'beds',
                                                'price',
                                                'minimum_nights',
                                                'availability_90',
                                                'availability_365',
                                                'number_of_reviews',
                                                'review_scores_rating',
                                                'review_scores_accuracy',
                                                'review_scores_cleanliness',
                                                'review_scores_checkin',
                                                'review_scores_communication',
                                                'review_scores_location',
                                                'review_scores_value',
                                                'instant_bookable'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model2.predict(input_variables)[0].round()

        return render_template('index.html',
                               original_input={'Nightly Rate': f"${price}",
                                               'Neighborhood': form_data['neighborhood_choice'],
                                               'Accomodates': accommodates,
                                               'Bedrooms': bedrooms,
                                               'Beds': beds,
                                               'Bathrooms': bathrooms},
                               result=(int(str(prediction).lstrip('[').rstrip('.]'))),
                               )

if __name__ == "__main__":
    app.run(debug=True)
