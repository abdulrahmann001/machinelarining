
# Numpy --------------------------------------------------

# Import numpy library for array operations
import numpy as np

# Create a 1D numpy array with values 1-4
arr = np.array([1,2,3,4])

# Display the array contents
arr

# Print the array using print()
print(arr)

# Create a mixed-type array with integers and strings
arr2 = np.array([1,2,3,'h','Soran'])

# Print the mixed-type array
print(arr2)

# Create array with range 0-49 using Python's range()
arr = np.array(range(50))

# Display the 0-49 array
arr

# Create array with values 5-19 using range
arr = np.array(range(5,20))

# Display the 5-19 array
arr

# Create array with odd numbers between 5-49
arr = np.array(range(5,50,2))

# Display the odd numbers array
arr

# Create array 0-9 using numpy.arange
arr=np.arange(10)

# Display arange array
arr

# Create array from 5-19 with step size 2 using arange
np.arange(5,20,2)

# Create 3x3 2D array
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

# Display 3x3 array
arr

# Create 3x4 2D array
arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# Display 3x4 array
arr

# Create vertical array with single-column structure
arr = np.array([[1],[2],[3],[5],[8],[9],[8]])

# Display vertical array
arr

# Transpose the vertical array
arr.T

# Create 3x4 array with alignment continuation
arr = np.array([[1,2,3,6],[4,5,6,6],[7,8,9,6]])

# Display 3x4 array
arr

# Transpose the 3x4 array
arr.T

# Transpose using transpose() method
arr.transpose()

# Create 10-element array of zeros
np.zeros(10)

# Create 5x3 matrix of zeros
np.zeros((5,3))

# Create 10-element array of ones
np.ones(10)

# Create 5x3 matrix of ones
np.ones((5,3))

# Create 4x4 identity matrix
np.eye(4)

# Create 50 evenly spaced values between 0-49
np.linspace(0,49,50)

# Generate 4 random floats between 0-1
np.random.rand(4)

# Generate 5x3 matrix of random floats
np.random.rand(5,3)

# Generate single random integer below 10
np.random.randint(10)

# Generate single random integer between 20-50
np.random.randint(20,50)

# Generate array of 10 random integers between 20-50
np.random.randint(20,50,10)

# Generate 4 samples from standard normal distribution
np.random.randn(4)

# Create array of 20 random integers between 2-50
arr=np.random.randint(2,50,20)

# Display random integer array
arr

# Check array dimensions with shape
arr.shape

# Reshape array to 4x5 matrix
arr.reshape(4,5)

# Store reshaped 5x4 matrix in variable b
b = arr.reshape(5,4)

# Display reshaped matrix
b

# Check shape of b
b.shape

# Reshape array to 20x1 column vector
c = arr.reshape(20,1)

# Display column vector
c

# Check column vector shape
c.shape

# Create new random integer array
arr=np.random.randint(2,50,20)

# Display new array
arr

# Reshape with automatic dimension calculation
arr.reshape(5,-1)

# Create larger random array and reshape
a =np.random.randint(2,400,80)
a.reshape(20,-1)

# Find maximum value in array
a.max()

# Find minimum value in array
a.min()

# Find index of maximum value
a.argmax()

# Find index of minimum value
a.argmin()

# Create 3x3 matrix and flatten it
d = np.array([[1,2,3],[3,4,5],[5,6,7]])
g = d.flatten()

# Display original and flattened arrays
d
g

# Compare shapes before/after flattening
d.shape
g.shape

# Create two 2x2 matrices and stack them
a = np.array([[1,2],[3,4]])
b = np.array([[4,5],[6,7]])
print(a)
print(b)

# Vertical stacking of matrices
np.vstack((a,b))

# Horizontal stacking of matrices
np.hstack((a,b))

# Alternative concatenation method
np.concatenate((a,b))

# Array indexing and slicing examples
a= np.random.randint(2,50,20)
a[10]
a[5:11]
a[:11]
a[5:]
a[::-1]  # Reverse array
a[::-2]  # Reverse with step 2
a[0:5]=100  # Modify first 5 elements
a[::]=300  # Set all elements to 300

# Demonstrate array view vs copy
a = np.random.randint(2,50,20)
slice_of_array = a[0:5]
slice_of_array[::]=100  # Modifies original array

a = np.random.randint(2,50,20)
slice_of_array = a.copy()[0:5]  # Independent copy
slice_of_array[::]=100  # Doesn't modify original

# 2D array indexing examples
z = np.random.randint(2,50,20).reshape(4,5)
z[2,3]  # Single element access
z[2][3]  # Alternative access method
z[2]  # Third row
z[:,1]  # Second column
z[0:3,0:2]  # Submatrix slice
z[1:3,1:4]  # Inner submatrix

# Advanced indexing techniques
a = np.random.randint(2,60,40).reshape(8,-1)
a[2:6]  # Row slice
a[:3]  # First three rows
a[:3][0]  # First row of first three
a[3:,2:][2,2]  # Chained indexing
a[[3,5,6]]  # Index list selection
a[[3,6],2:]  # Row and column selection
a[[3,5],:][:,[1,2,4]]  # Double selection

# Boolean indexing examples
a = np.random.randint(2,50,20)
a>10  # Boolean mask
a[a>10]  # Filter using mask
a[a!=10]  # Exclusion filter
a[a%2==0]  # Even numbers filter

# Array arithmetic operations
a = np.arange(2,10)
b = np.arange(3,11)
a+b  # Element-wise addition
a-b  # Element-wise subtraction
a*3  # Scalar multiplication
a+50  # Scalar addition

# Mathematical functions applied to arrays
a =np.random.randint(2,50,30)
np.sqrt(a)  # Square roots
np.exp(a)  # Exponentials
np.cos(a)  # Cosines
np.sin(a)  # Sines
np.max(a)  # Maximum value
np.min(a)  # Minimum value
np.log(a)  # Natural logarithms



# Pandas --------------------------------------------------

# Import essential data analysis libraries
import numpy as np
import pandas as pd

# Create sample data structures for Series demonstration
index1 = ['a', 'b','c']
my_list = [1,2,3]
arr = np.array([10,20,30])
d = {'a':20, 'b':50,'c':60}

# Create basic Series from list with default integer index
s1 = pd.Series(my_list)
s1

# Access first element using positional index
s1[0]

# Create Series with custom string index
s2 = pd.Series(my_list, index = index1)
s2

# Access element using custom index label
s2['a']

# Access element using both label and position
s2[0]

# Create Series from NumPy array
s3 = pd.Series(arr)
s3

# Create Series from dictionary (keys become index)
s4 = pd.Series(d)
s4

# Create Series with dictionary data but new index (creates NaN)
index2 = ['i','j','k']
s5 = pd.Series(d, index = index2)
s5

# Demonstrate Series alignment during operations
s1 = pd.Series(data = [10,20,30,40], index = ['a','b','c','d'])
s2 = pd.Series(data = [1,2,3,4], index = ['a','e','c','f'])
s1+s2  # Aligns on index, fills missing with NaN

# Create DataFrame with random data and custom labels
df = pd.DataFrame(data = np.random.randn(5,4), 
                 index = ['a','b','c','d','e'], 
                 columns = ['x','y','z','w'])
df

# Check DataFrame type
type(df)

# Access single column as Series
df['x']
type(df['x'])

# Access multiple columns as DataFrame
df[['x','z']]

# Add new calculated column to DataFrame
df['new'] = df['x'] + df['y']
df

# Add Series as new column (index alignment)
s3 = pd.Series(data = [1,2,3,4,5], index = ['a','b','c','d','e'])
df['payam'] = s3
df

# Demonstrate column removal (temporary)
df.drop('payam', axis=1)

# Permanently remove column using inplace
df.drop('payam', axis=1, inplace=True)

# Label-based indexing with loc
df.loc['a']  # Get row by index label
df.loc['a']['x']  # Get specific cell
df.loc['a'][['x','y']]  # Get multiple columns
df.loc[['a','b']]  # Multiple rows
df.loc[['a','b','e']][['x','w']]  # Combined selection

# Position-based indexing with iloc
df.iloc[0]  # First row

# Boolean indexing and filtering
df > 0  # Boolean DataFrame
df[df > 0]  # Filter values
df[df['w'] > 0]  # Filter rows by column condition
df[(df['w'] > 0) & (df['z'] > 0)]  # Multiple conditions
df[(df['w'] > 0) | (df['z'] > 0)]

# Index manipulation examples
df.reset_index()  # Convert index to column
df.reset_index(inplace=True)  # Permanent reset
df.drop('index', axis=1, inplace=True)  # Remove index column

# Set new index from column
new_index = "tatar shad aram darya mustafa".split()
df['new_index'] = new_index
df.set_index('new_index', inplace=True)

# Create hierarchical MultiIndex DataFrame
outside = ['G1', 'G1', 'G1', 'G2', 'G2']
inside = [1,2,3,1,2]
heir_index = pd.MultiIndex.from_tuples(list(zip(outside, inside)))
df = pd.DataFrame(np.random.randn(5,4), index=heir_index, columns=['x','y','z','w'])
df.loc['G1']  # Access outer index level

# Missing data handling demonstration
df = pd.DataFrame({'A':[1,2,np.nan], 'B':[5, np.nan, np.nan], 'C':[1,2,3]})
df.isnull()  # Boolean null mask
df.isnull().sum()  # Column-wise null counts
df.dropna()  # Remove rows with nulls
df.dropna(axis=1)  # Remove columns with nulls
df.dropna(thresh=2)  # Keep rows with at least 2 non-nulls
df.fillna(value = df['A'].mean())  # Fill nulls with column mean

# GroupBy operations and aggregation
data = {'company':['Google', 'Google', 'Microsoft', 'Microsoft', 'Korek', 'Korek'],
        'person':['Ashna', 'shad', 'aram', 'Saeed', 'shaho', 'chiman'],
        'sales':[200, 1000, 350, 500, 800, 2500]}
df = pd.DataFrame(data)
df.groupby('company').mean()  # Group averages
df.groupby('company').agg({'sales':['count','max', 'std', 'mean']})

# Data cleaning and transformation
df = pd.DataFrame({'col1':[1,2,3,4], 'col2':[20,20,40,50], 'col3':['a','b','c','d']})
df['col2'].unique()  # Distinct values
df['col2'].value_counts()  # Value frequencies
df['col1'].apply(lambda x: x*2)  # Apply function to column

# File I/O operations
df = pd.read_csv("example1.csv")  # Read CSV file
df.drop('Unnamed: 0', axis=1, inplace=True)  # Clean index column
df.to_csv("rediyar.csv")  # Write to CSV
df.to_excel("mustafa.xlsx")  # Write to Excel
