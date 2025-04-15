import numpy as np

class OutlierDetection :
    """Classe utilitaire implémentant des méthodes pour détections des outliers ou données aberrantes
    """
    def __init__(self, data):
        """
        Initialise le détecteur d'outliers
        Args:
            data: données
        """
        self.data_dir = data
        
    def calculate_iqr(self):
        """
        Calculate the Interquartile Range (IQR) for a given dataset.
        
        Parameters:
        data (array-like): The dataset for which to calculate the IQR
        
        Returns:
        tuple: A tuple containing (Q1, Q3, IQR, lower_bound, upper_bound)
        """
        # Convert to numpy array if not already
        data_array = np.array(self.data)
        
        # Calculate quartiles
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        
        # Calculate IQR
        iqr = q3 - q1
        
        # Calculate bounds for outlier detection
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return q1, q3, iqr, lower_bound, upper_bound
    
    def identify_outliers_by_iqr(self):
        """
        Identify outliers in a dataset using the IQR method.
        
        Parameters:
        data (array-like): The dataset to check for outliers
        
        Returns:
        tuple: A tuple containing (outliers, is_outlier_mask)
        """
        data_array = np.array(self.data)
        q1, q3, iqr, lower_bound, upper_bound = self.calculate_iqr(data_array)
        
        # Create mask for outliers
        is_outlier = (data_array < lower_bound) | (data_array > upper_bound)
        
        # Get the outliers
        outliers = data_array[is_outlier]
        
        return outliers, is_outlier
       