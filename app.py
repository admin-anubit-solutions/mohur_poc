import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit import multiselect, select_slider, selectbox
import ast
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from streamlit_modal import Modal
import streamlit.components.v1 as components

def get_recommendations(user_data, top_n=10):
	print(user_data)
	def flatten(x):
		import ast
		x = x.fillna('[]')
		data = f"{x['country']} {x['state']} {x['city']} {' '.join(ast.literal_eval(x['industries']))} {' '.join(ast.literal_eval(x['sectors']))} {' '.join(ast.literal_eval(x['stages']))}"
		return data
	
	def flatten_user(x):
		# flat_data = f"{data['country']} {data['state']} {data['city']} {' '.join(data['industries'])} {' '.join(data['sectors'])} {' '.join(data['stages'])}"
		# return flat_data
		import ast
		data = f"{x['country']} {x['state']} {x['city']} {' '.join(ast.literal_eval(x['industries']))} {' '.join(ast.literal_eval(x['sectors']))} {' '.join(ast.literal_eval(x['stages']))}"
		return data

	# Load item dataset
	item_dataset = pd.read_csv('Combined_Funding_utf.csv')

	item_features = ['country', 'state', 'city', 'industries', 'sectors', 'stages']
	item_dataset['item_text'] = item_dataset[item_features].apply(flatten, axis=1)
	# Get user profile
	user_data_df = pd.DataFrame(user_data, index=[0])
	user_data_df['user_text'] = flatten_user(user_data)

	vectorizer = TfidfVectorizer()
	item_vectors = vectorizer.fit_transform(item_dataset['item_text'])
	user_vector = vectorizer.transform(user_data_df['user_text'])

	user_profile = user_vector.mean(axis=0)
	user_profile_array = np.array(user_profile)
	item_array = item_vectors.toarray()

	similarity_scores = cosine_similarity(user_profile_array, item_array)

	top_items_indices = similarity_scores.argsort()[0][::-1][:top_n]
	recommended_items = item_dataset.iloc[top_items_indices]

	return recommended_items

def get_recommendations_hybrid(user_data, top_n=10):

	def flatten_ind_sec(input):
		try:
			parsed_ind = ast.literal_eval(input['industries'])
		except:
			parsed_ind = []
		result_ind = ' '.join(parsed_ind)
		try:
			parsed_sec = ast.literal_eval(input['sectors'])
		except:
			parsed_sec = []
		result_sec = ' '.join(parsed_sec)
		return result_ind + ' ' + result_sec
	
	def flatten_stage(input):
		try:
			parsed_stage = ast.literal_eval(input['stages'])
		except:
			parsed_stage = []
		result_stage = ' '.join(parsed_stage)
		return result_stage
	
	def set_geo(item_dataset):
		geography = item_dataset['city'] + ' ' + item_dataset['state'] + ' ' + item_dataset['country']
		return geography
	
	def flatten_user(input):
		parsed_ind = ast.literal_eval(input['industries'])
		result_ind = ' '.join(parsed_ind)
		parsed_sec = ast.literal_eval(input['sectors'])
		result_sec = ' '.join(parsed_sec)
		# parsed_stage = ast.literal_eval(input['stages'])
		# result_stage = ' '.join(parsed_stage)
		return result_ind + ' ' + result_sec + ' ' + input['stages'] + input['city'] + ' ' + input['state'] + ' ' + input['country']
	
	# Load item dataset
	item_dataset = pd.read_csv('data.csv')
	item_dataset = item_dataset.fillna('')
	item_dataset = item_dataset[item_dataset['stages'].str.contains(user_data['stages'], case=False)]
	# Define weights for 'industry_sector', 'stages', and 'item_text'
	industry_sector_weight = 3.0  # Adjust this weight as needed
	stages_weight = 2.0  # Adjust this weight as needed
	item_text_weight = 1.0  # Adjust this weight as needed

	# Custom vectorizer to apply weights consistently for both item and user vectorization
	class CustomTfidfVectorizer(TfidfVectorizer):
		def transform(self, raw_documents):
			X = super().transform(raw_documents)
			return X

	vectorizer_industry_sector = CustomTfidfVectorizer()
	vectorizer_stages = CustomTfidfVectorizer()
	vectorizer_item_geo = CustomTfidfVectorizer()
	
	# Concatenate 'industries' and 'sectors' into 'industry_sector' and store it in 'industry_sector'
	# df['Column1'] + df['Column2'].astype(str)
	item_dataset['industry_sector'] = item_dataset['industries'] + item_dataset['sectors']
	item_dataset['geography'] = item_dataset['country'] + item_dataset['state'] + item_dataset['city']
	# Fit and transform 'industry_sector', 'stages', and 'item_text' separately with their respective weights
	item_vectors_industry_sector = vectorizer_industry_sector.fit_transform(item_dataset['industry_sector'])
	item_vectors_stages = vectorizer_stages.fit_transform(item_dataset['stages'])
	item_vectors_item_geo = vectorizer_item_geo.fit_transform(item_dataset['geography'])
	from scipy.sparse import hstack
	from scipy.sparse import csc_matrix

	# Find the maximum number of columns among the matrices
	max_columns = max(item_vectors_industry_sector.shape[1], item_vectors_stages.shape[1], item_vectors_item_geo.shape[1])

	# Define a function to pad a matrix with zeros to match the maximum number of columns
	def pad_matrix(matrix, max_columns):
		if matrix.shape[1] < max_columns:
			num_columns_to_add = max_columns - matrix.shape[1]
			zero_columns = csc_matrix((matrix.shape[0], num_columns_to_add))
			matrix = hstack([matrix, zero_columns], format='csc')
		return matrix

	# Pad the matrices
	item_vectors_industry_sector_padded = pad_matrix(item_vectors_industry_sector, max_columns)
	item_vectors_stages_padded = pad_matrix(item_vectors_stages, max_columns)
	item_vectors_item_geo_padded = pad_matrix(item_vectors_item_geo, max_columns)

	# Combine the weighted representations of 'industry_sector', 'stages', and 'item_text'
	item_vectors_combined = (industry_sector_weight * item_vectors_industry_sector_padded +
							stages_weight * item_vectors_stages_padded +
							item_text_weight * item_vectors_item_geo_padded)

	# Apply the same weight to the 'stages' feature for user vectorization
	user_data_df = pd.DataFrame(user_data, index=[0])
	user_data_df['user_data'] = user_data_df.apply(flatten_user, axis=1)

	user_vector = vectorizer_item_geo.transform(user_data_df['user_data'])  # Use the 'item_text' vectorizer for user vectorization
	user_vector_padded = pad_matrix(user_vector, max_columns)
	user_profile = user_vector_padded.mean(axis=0)
	user_profile_array = np.array(user_profile)
	item_array = item_vectors_combined.toarray()

	similarity_scores = cosine_similarity(user_profile_array, item_array)

	top_items_indices = similarity_scores.argsort()[0][::-1][:top_n]
	recommended_items = item_dataset.iloc[top_items_indices]

	return recommended_items

def main():
	st.set_page_config(
	layout="centered", page_icon="🖱️", page_title="MOHUR PoC"
)
	st.title("MOHUR Recommendation System PoC")

	# User input
	# country = st.text_input("Enter country")
	state = st.selectbox(label="Enter state",options=[
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
    'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
    'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
    'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
    'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
    'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu',
    'Lakshadweep', 'Delhi', 'Ladakh', 'Lakshadweep', 'Puducherry'
],index=10)
	city = st.text_input("Enter city",value='Bengaluru')
	# industries = st.text_input("Enter industries (separated by commas)")
	# sectors = st.text_input("Enter sectors (separated by commas)")
	# stages = st.text_input("Enter stages (separated by commas)")
	industries = multiselect("Select industries", options=['Retail', 'Biotechnology', 'Textiles & Apparel', 'Art & Photography', 'Food & Beverages', 'Robotics', 'Dating Matrimonial', 'Internet of Things', 'Telecommunication & Networking', 'Automotive', 'Events', 'Aeronautics Aerospace & Defence', 'Toys and Games', 'Analytics', 'Design', 'Healthcare & Lifesciences', 'House-Hold Services', 'Logistics', 'Computer Vision', 'Social Network', 'Professional & Commercial Services', 'Advertising', 'Finance Technology', 'Safety', 'Fashion', 'Human Resources', 'Green Technology', 'Airport Operations', 'AI', 'Agriculture', 'Enterprise Software', 'IT Services', 'Security Solutions', 'Real Estate', 'Technology Hardware', 'Travel & Tourism', 'Chemicals', 'Passenger Experience', 'Transportation & Storage', 'AR VR (Augmented + Virtual Reality)', 'Construction', 'Animation', 'Non- Renewable Energy', 'Indic Language Startups', 'Nanotechnology', 'Sports', 'Social Impact', 'Marketing', 'Waste Management', 'Architecture Interior Design', 'Others', 'Education', 'Renewable Energy', 'Other Specialty Retailers', 'Pets & Animals', 'Media & Entertainment'])
	sectors = multiselect("Select sectors", options=['Business Intelligence', 'Art', 'Recruitment Jobs', 'Healthcare Services', 'Oil Related Services and Equipment', 'Wearables', 'Aviation & Others', 'Digital Marketing (SEO Automation)', 'Enterprise Mobility', 'Natural Language Processing', 'Public Citizen Security Solutions', 'Application Development', 'Animal Husbandry', 'Food Processing', 'Digital Media Blogging', 'Crowdfunding', 'Construction Supplies & Fixtures', 'Product Development', 'Renewable Nuclear Energy', 'Testing', 'Biotechnology', 'Big Data', 'Fashion Technology', 'Others', 'Apparel', 'Coworking Spaces', 'Education', 'Web Design', 'Home Improvement Products & Services Retailers', 'Computer & Electronics Retailers', 'Apparel & Accessories', 'Business Finance', 'Advisory', 'Robotics Technology', 'Education Technology', 'Oil & Gas Drilling', 'Environmental Services & Equipment', 'Skill Development', 'Project Management', 'Digital Media', 'Holiday Rentals', 'Cloud', 'Comparison Shopping', 'Hotel', 'Bitcoin and Blockchain', 'Fisheries', 'Personal Security', 
'OOH Media', 'Media and Entertainment', 'AdTech', 'Utility Services', 'Freight & Logistics Services', 'Customer Support', 'Drones', 'Data Science', 'Transport Infrastructure', 'Baby Care', 'Accounting', 'Food Technology/Food Delivery', 'Healthcare Technology', 'Retail Technology', 'Leather Textiles Goods', 'Handicraft', 'Specialty Chemicals', 'Online Classified', 'Microbrewery', 'Commercial Printing Services', 'Restaurants', 'Cyber Security', 'Housing', 'Event Management', 'Embedded', 'Clean Tech', 'Horticulture', 'Renewable Wind Energy', 'Assistance Technology', 'Corporate Social Responsibility', 'Location Based', 'Experiential Travel', 'Entertainment', 'Electric Vehicles', 'BPO', 'Digital Media Video', 'Oil & Gas Exploration and Production', 'Market Research', 'Wireless', 'Construction Materials', 'KPO', 'Skills Assessment', 'Dairy Farming', 'Oil & Gas Transportation Services', 'Agricultural Chemicals', 'ERP', 'Social Commerce', 'Photography', 'Trading', 'Microfinance', 'Billing and Invoicing', 'Agri-Tech', 'Discovery', 'Network Technology Solutions', 'Organic Agriculture', 'Professional Information Services', 'Insurance', 'NGO', 'Business Support Services', 'Sports Promotion and Networking', 'Robotics Application', 'Web Development', 'Hospitality', 'Diversified Chemicals', 'Training', 'Space Technology', 'Sales', 'Renewable Energy Solutions', 'Home Furnishings Retailers', 'Personal Finance', 'Smart Home', 'New-age Construction Technologies', 'NLP', 'Fan Merchandise', 'Manufacture of Electrical Equipment', 'Health & Wellness', 'Tires & Rubber Products', 'Lifestyle', 'E-learning', 'Non- Leather Footwear', 'SCM', '3d printing', 'Non- Leather Textiles Goods', 'Wayside Amenities', 'Ticketing', 'Branding', 'Healthcare IT', 'IT Management', 'Manufacturing & Warehouse', 'Coaching', 'Payment Platforms', 'E-Commerce', 'Homebuilding', 'Digital Media Publishing', 'Waste Management', 'Medical Devices Biomedical', 'Construction & Engineering', 'Traffic Management', 'Mobile wallets  Payments', 'Defence Equipment', 'Facility Management', 'Talent Management', 'P2P Lending', 'Integrated communication services', 'Industrial Design', 'CXM', 'Employment Services', 'Physical Toys and Games', 'Jewellery', 'Foreign Exchange', 'Digital Media News', 'Manufacturing', 'Virtual Games', 'Social Media', 'Internships', 'Business Support Supplies', 'Pharmaceutical', 'Leather Footwear', 'Commodity Chemicals', 'Point of Sales', 'Manufacture of Machinery and Equipment', 'Machine Learning', 'Collaboration', 'Home Security solutions', 'Auto Vehicles, Parts & Service Retailers', 'Passenger Transportation Services', 'Renewable Solar Energy', 'Auto & Truck Manufacturers', 'Weddings', 'IT Consulting', 'Movies', 'Fantasy Sports', 'Electronics', 'Loyalty', 'Home Care', 'Auto, Truck & Motorcycle Parts', 'Personal Care', 'Laundry', 'Semiconductor'])
	stages = select_slider("Select your stage", options=['Prototype', 'Validation', 'EarlyTraction', 'Scaling'])
	print(f'stages: {stages}')
	# if st.button("Get Recommendations"):
	# 	user_data = {
	# 		'country': country,
	# 		'state': state,
	# 		'city': city,
	# 		'industries': str(industries),
	# 		'sectors': str(sectors),
	# 		'stages': str(stages)
	# 	}
	# 	print(f'USER DATA: {user_data}')
	# 	recommendations = get_recommendations(user_data, top_n=5)
		# st.write(recommendations)

	if st.button("Get Recommendations"):
		user_data = {
			'country': 'India',
			'state': state,
			'city': city,
			'industries': str(industries),
			'sectors': str(sectors),
			'stages': str(stages)
		}
		# print(f'USER DATA: {user_data}')
		recommendations = get_recommendations_hybrid(user_data, top_n=10)
		for index, row in recommendations.iterrows():
				data = [ 'name', 'role', 'description', 'stages', 'sectors', 'industries', 'city', 'state', 'country']
				addon = [ 'budget', 'ministry', 'department', 'portfolios']
				for entry in addon:
					if row[entry]:
						data.append(entry)
				with st.expander(f"{row['name']} | {row['role']}"):
					if 'image' in row:
						st.image(f"https://api.startupindia.gov.in/sih/api/file/user/image/Accelerator?fileName={row['image']}", width=300)
					if 'budget' in row and row['budget'].split('_') != ['']:
						budget = row['budget']
						temp = budget.split('_')
						# print(temp)
						budget = f'{temp[1]} {temp[0]}'
						if len(temp) <= 3:
							budget += ' and above'
						else:
							budget += f' - {temp[3]} {temp[2]}'
						row['budget'] = budget
					st.write(row[data])
					# Display the 'contacts' information
					if 'contacts' in row and row['contacts'] != '':
						print('dfvdsffds, ',row['contacts'])
						contacts_data = ast.literal_eval(row['contacts'])

						if contacts_data:
							st.subheader("Contacts:")
							for contact_info in contacts_data:
								st.write(f"**Name:** {contact_info['firstName']} {contact_info['lastName']}")
								st.write(f"**Designation:** {contact_info['designation']}")
								st.write(f"**Email:** {contact_info['emailId']}")
								st.write(f"**Mobile Number:** {contact_info['mobileNumber']}")
								st.write(f"**Landline Number:** {contact_info['landlineNumber']}")
								st.write(f"**Website:** {contact_info['website']}")
								st.write(f"**Social Media URL:** {contact_info['socialMediaAccountURL']}")





if __name__ == "__main__":
	main()
