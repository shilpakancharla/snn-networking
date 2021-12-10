import boto3
import random
import pandas as pd 
from io import StringIO
from sagemaker import get_execution_role

"""
    Author: Shilpa Kancharla
    Last Modified: December 9, 2021
"""

"""
    Accessing data from bnncdata S3 bucket.
    
    @param resource: S3 resource object
    @param bucket: name of S3 bucket
    @param subfolder_name: subfolder name within S3 bucket
    @return list of filepaths associated with a subfolder
"""
def get_filenames_from_subfolder(resource, bucket, subfolder_name):
    filepath_list = []
    bucket_obj = resource.Bucket(bucket)
    for object_summary in bucket_obj.objects.filter(Prefix = subfolder_name):
        filepath_list.append(object_summary.key)
    return filepath_list

"""
    Accessing data and creating dataframe from S3 bucket.
    
    @param client: S3 client object
    @param bucket: name of S3 bucket
    @param key: filepath within S3 bucket
    @return dataframe of requested object
"""
def create_df(client, bucket, key):
    obj = client.get_object(Bucket = bucket, Key = key)
    df = pd.read_csv(obj['Body'])
    return df

"""
    Concatenate dataframes together. 
    
    @param df_list: list of dataframes to concatenate together
    @return concatenated dataframes
"""
def concatenate_df(df_list):
    concatenated_df = pd.concat(df_list, ignore_index = True)
    return concatenated_df

"""
    Organize the files accordint to whether they belong to the training, validation, or
    test sections.
    
    Training: 25, 30, 35, 40, 45, and 50
    Validation and test: 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 
        150, 160, 170, 180, 190, 200, 220, 240, 260, 280, and 300
        
    @param subfolders: list of subfolders in S3 bucket
    @param resource: S3 resource object
    @param bucket: name of S3 bucket
    @return list of training files in S3 bucket
    @return list of validation files in S3 bucket
    @return list of test files in S3 bucket
"""
def train_validation_test_split(subfolders, resource, bucket):
    training_nodes = [25, 30, 35, 40, 45, 50]
    val_test_nodes = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 
                        150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300]
    training_files = []
    validation_files = []
    test_files = []
    
    for s in subfolders:
        for node in training_nodes:
            item = 'tabular_data/' + str(node) + '/'
            if item in s:
                filepath_list = get_filenames_from_subfolder(resource, bucket, s)
                for f in filepath_list:
                    training_files.append(f)
        for node in val_test_nodes:
            item = 'tabular_data/' + str(node) + '/'
            if item in s:
                filepath_list = get_filenames_from_subfolder(resource, bucket, s)
                for f in filepath_list:
                    if 'test' in f:
                        test_files.append(f)
                    else:
                        validation_files.append(f)
                
    return training_files, validation_files, test_files

"""
    Create a concatenated dataframe based on the number of nodes and save it as either a training, validation, or
    test file. 
    
    @param section_of_data: list of either training, validation, or test files
    @param node: number of nodes in topology
    @param bucket: S3 bucket object
    @param client: S3 client object
    @param resource: S3 resource object
    @return concatenated dataset for that particular node
"""
def create_list_of_frames(section_of_data, node, bucket, client):
    output_frame_list = []
    count = 0
    for file in section_of_data:
        if count < 5: # Limit so dead kernel or gateway timeout does not occur
            item = 'results_' + str(node)
            if item in file:
                _df = create_df(client, bucket, file)
                output_frame_list.append(_df)
                count = count + 1
                print("Added " + file + " to output frame list.")
                print("Count: " + str(count))

    # Do random sampling if the list of dataframes is too long
    if len(output_frame_list) > 50:
        print("Create random sample of list.")
        output_frame_list = random.choices(output_frame_list, k = 50)
        print("Finished creating random sample of the list of length " + str(len(output_frame_list)) + ".")
    
    return output_frame_list

"""
    Save a concatenated dataframe in the S3 bucket.
    
    @param df_list: list of dataframes
    @param mode: select either training, validation, or test as a string input 
    @param node: number of nodes in topology
    @param bucket: S3 bucket object
    @param resource: S3 resource object
"""
def save_df_to_s3(df_list, mode, node, bucket, resource):
    print("Concatenating dataframes.")
    df = concatenate_df(df_list)
    print("Finished concatenating dataframes.")
    # Save to S3 bucket
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    if mode == 'training':
        resource.Object(bucket, 'training_' + str(node) + '.csv').put(Body = csv_buffer.getvalue())
        print("Finished creating .csv in S3 bucket.")
    elif mode == 'validation':
        resource.Object(bucket, 'validation_' + str(node) + '.csv').put(Body = csv_buffer.getvalue())
        print("Finished creating .csv in S3 bucket.")
    elif mode == 'test':
        resource.Object(bucket, 'test_' + str(node) + '.csv').put(Body = csv_buffer.getvalue())
        print("Finished creating .csv in S3 bucket.")

# Driver code
if __name__ == "__main__":
    role = get_execution_role()

    bucket = 'bnncdata'
    prefix = 'tabular_data/'

    # Retrieve subfolder names in S3 bucket
    subfolders = []
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    result = client.list_objects(Bucket = bucket, Prefix = prefix, Delimiter = '/')
    for obj in result.get('CommonPrefixes'):
        subfolders.append(obj.get('Prefix'))

    # Get the filepaths for the training, validation, and test sets
    training_files, validation_files, test_files = train_validation_test_split(subfolders, resource, bucket)

    #test_frames_300 = create_list_of_frames(test_files, 300, bucket, client)
    #save_df_to_s3(test_frames_300, 'test', 300, bucket, resource)