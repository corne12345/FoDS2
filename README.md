Steps to follow:
- Data Cleaning
- Selecting relevant features
- Feature engineering (making features based on other features)

Interesting features:
- Orientation (image_height/image_width)
- Image filter (one-hot encoding or classification according to media)
- Data memorability
- User_bio
- User_follows, User_followed_by
- Popularity (user_followed_by / user_follows)
- user_posted_photos
- anp_sentiment, label (beredening voor ordinaliteit nagaan?)
- emotion_score, label (beredenering voor ordinaliteit nagaan?)
- data_amz_label, data_amz_confidence (alles boven x% confidence, onderverdelen in categoriën)
- Face_gender (gewogen voor confidence)
- face_age_mean (face_age_range_high + face_age_range_low / 2)
- face_sunglasses, face_beard, face_mustache, ...., emo_confidence
- comment_count
- like_count
- gender 
- born
- education
- employed
- income (cleaning nodig)
- HAP
- participate
- end_q - start_q 
- imagecount
- private_account boeit niet, want heeft maar 1 waarde

Options:
- Faces (minder samples, maar meer parameters), op basis van data_amz_label
- Alle (meer samples, maar niet alle parameters)
- Perma score valideren





