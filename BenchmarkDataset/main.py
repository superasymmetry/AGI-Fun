from iterators import define_iterator, classify_iterator, generate_iterator, edit_iterator
from potemkin_rates import print_potemkin_rate_by_task
from pprint import pprint


# Example usage of the four main iterators
if __name__ == '__main__':
    # Example 1: Define task
    iterator = define_iterator()   
    row, inference = next(iterator) # Get the first (row, inference) pair from the iterator
    # print("First row in the define iterator:")
    # pprint(row)    
    # print(inference) # Uncomment to print the corresponding inference text


    # Example 2: Classify task
    iterator = classify_iterator()
    row, inference = next(iterator)
    # print("First row in the classify iterator:")
    # pprint(row)    
    # print(inference) # Uncomment to print the corresponding inference text


    # Example 3: Generate task
    iterator = generate_iterator()
    row, inference = next(iterator)
    # print("First row in the generate iterator:")
    # pprint(row)    
    # print(inference) # Uncomment to print the corresponding inference text


    # Example 4: Edit task
    iterator = edit_iterator()
    row, inference = next(iterator)
    # print("First row in the edit iterator:")
    # pprint(row)
    # print(inference) # Uncomment to print the corresponding inference text


    # Print potemkin rates (reproducing Table 1)
    print_potemkin_rate_by_task()
