### knn_fxns.py
import numpy as np
import pandas as pd

#first two functions written locally

def row_distance(row1, row2):
    '''
    Returns the distance between two input rows, extracted from a Pandas dataframe
    INPUT: two rows which should be Pandas series or array-type, not data frame 
    OUTPUT: Euclidean disstance
    '''
    arr1=np.array(row1)
    arr2=np.array(row2)
    distance=np.sqrt(sum((arr1-arr2)**2))
    return distance
    

def calc_distance_to_all_rows(df,example_row ):
    '''
    Computes distance between every row in input df (Pandas dataframe) and example_row (Pandas series or array type)
    Calls 'row_distance'
    INPUT: df, Pandas dataframe; example_row
    OUTPUT:Pandas dataframe with additional column 'distance_to_ex' added to input dataframe df
    '''
    distances=[]
    drop_df=df.drop(['class'],axis=1)
    rows=len(drop_df)
    for i in range(0,rows):
        curr_row=drop_df.iloc[i,:]
        dist=row_distance(curr_row, example_row)
        distances.append(dist)
    new_df=df.assign(distance_to_ex=distances)
    return new_df
    

def find_k_closest(df, example_row, k):
    """
    Finds the k closest neighbors to example, excluding the example itself.
    Calls 'calc_distance_to_all_rows'
    IF there is a tie for kth closest, choose the final k to include via random choice.
    INPUT: df, Pandas dataframe; example_row, Pandas series or array type; k, integer number of nearest neighbors.
    OUTPUT: dataframe in same format as input df but with k rows and sorted by 'distance_to_ex.'
    """
    k_df=calc_distance_to_all_rows(df,example_row)
    sorted_df=k_df.sort_values(by=['distance_to_ex'])
    #if(sorted_df.i[k]==sorted_df.i[k+1]):
    return sorted_df[0:k]
        
                     
                
    
    
def classify(df, example_row, k):
    """
    Return the majority class from the k nearest neighbors of example
    Calls 'find_k_closest'
    INPUT: df, Pandas dataframe; example_row, Pandas series or array type; k, integer number of nearest neighbors
    OUTPUT: string referring to closest class.
    """
    new_k_df=find_k_closest(df,example_row,k)
    majority=new_k_df['class'].max()
    return majority
    
