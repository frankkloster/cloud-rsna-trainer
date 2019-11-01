from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras>=2.2',
                     'h5py==2.7.0',
                     'pydicom>=1.3',
                     'scikit-learn>=0.21',
                     'opencv-python>=4',
                     'pandas>=0.23',
                     'gcsfs>=0.3'
                     ]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='RSNA CNN application'
)