### create a function of a seaborn piechart, the input would be a list of string representing the category and a list of int representing the value of each category
### the output would be a piechart with the percentage of each category, the title of the piechart would be the total number of the categories
### the piechart would be saved in a directory if the savefig is true and the directory is specified
### the piechart will be beautiful with seaborn package

def piechart(categories = [],
             values = [],
             savefig = False,
             directory = None, 
             percentage = False,
             figure_name = None):
    '''
    categories: a list of string representing the category
    values: a list of int representing the value of each category
    savefig: a boolean value, if true, save the piechart
    directory: a string representing the directory to save the piechart
    percentage: a boolean value, if true, the values will be converted to percentage
    figure_name: a string representing the name of the figure
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np
    import pandas as pd

    # check if the number of categories and values are the same
    assert len(categories) == len(values) , "The number of categories and values should be the same"
    if percentage:
        # check if the sum of values is 100
        assert sum(values) == 100 , "The sum of values should be 100"
        # convert the values to percentage
        values = [i/100 for i in values]
        # create a dataframe
        df = pd.DataFrame({'categories':categories,'values':values})
        # create a piechart
        plt.figure(figsize=(10,10))
        plt.pie(df['values'],labels=df['categories'],autopct='%1.1f%%')
        plt.title('Total number of categories: {}'.format(len(categories)))
        plt.show()
        # save the piechart
        if savefig:
            if directory == None:
                directory = os.getcwd()
            if figure_name == None:
                plt.savefig(directory + '/piechart.png')
            else:
                plt.savefig(directory + '/' + figure_name + '.png')
    else:
        #if it is not a percentage plot, then plot the values directly
        # create a dataframe
        df = pd.DataFrame({'categories':categories,'values':values})
        # create a piechart
        plt.figure(figsize=(5,5))
        plt.pie(df['values'],labels=df['categories'])
        plt.title(f"Total number of categories: {len(categories)}\nTotal number of cases {sum(values)}")
        
        plt.show()
        # save the piechart
    if savefig:
        if directory == None:
            directory = os.getcwd()
        if figure_name == None:
            plt.savefig(directory + '/piechart.png')
        else:
            plt.savefig(directory + '/' + figure_name + '.png')
            
            


### build a function, the input is a image dir, the output is the cropped images
### the orginal image is 96 * 96 *3, the output should be 64 * 64 * 3, center cropped
### the output should be saved in a new dir
import os
import cv2
import numpy as np


def intersection_of_two_lists(list1, list2):
    '''
    :param list1: list1
    :param list2: list2
    :return: the intersection of list1 and list2
    '''
    return list(set(list1).intersection(set(list2)))

def barplot(list_of_values = None,
            list_of_names = None,
            title = None,
            reference_value = None,
            color = None,):
    '''
    :param list_of_values: a list of values
    :param list_of_names: a list of names
    :param title: the title of the plot
    :param reference_value: the reference value, which will be a dash line
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    assert list_of_names is not None, 'list_of_names is None'
    assert list_of_values is not None, 'list_of_values is None'
    if color is None: # if color is None, then use the default color
        color = 'b'
    else:
        assert color in ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'], 'the color is not in the list'
    assert len(list_of_names) == len(list_of_values), 'the length of list_of_names is not equal to the length of list_of_values'
    plt.figure(figsize=(10, 10))
    print("color is not abailable yet")

    plt.bar(np.arange(len(list_of_values)), list_of_values, width=0.5, align='center')  # A bar chart
    plt.xticks(np.arange(len(list_of_values)), list_of_names)  # Change the xticks' name
    try:
        plt.axhline(reference_value, color='r', linestyle='--')
    except:
        pass
    try:
        plt.title(title)
    except:
        pass

