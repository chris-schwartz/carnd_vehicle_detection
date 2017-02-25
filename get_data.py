import requests, zipfile, io
vehicle_data_url = 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip'
non_vehicle_data_url = 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip'

vehicle_data_file = "vehicles.zip"
non_vehicle_data_file = "nonvehicles.zip"

print("Downloading and extracting all training data...\n")

print("downloading and extracting vehicle data...")
r = requests.get(vehicle_data_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
print("Finished!\n")

print("downloading and extracting non vehicle data...")
r = requests.get(non_vehicle_data_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
print("Finished!\n")

print("All training data downloaded and extracted!")

