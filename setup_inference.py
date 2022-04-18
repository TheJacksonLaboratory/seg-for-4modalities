from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Mouse brain segmentation for four modalities - inference only'
LONG_DESCRIPTION = 'Mouse brain segmentation for four modalities - inference only. \
                    Segments anatomical, DTI, NODDI, and fMRI image modalities. \
                    For full documentation, please see: \
                    https://github.com/TheJacksonLaboratory/seg-for-4modalities'

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
        name="seg-for-4modalities", 
        version=VERSION,
        url='https://www.jax.org',
        author="Zachary Frohock",
        author_email="zachary.frohock@jax.org",
        license='BSD 2-clause',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=required,        
        include_package_data=True,
        package_data={
            'seg-for-4modalities.predict.scripts': ['predict/scripts/*.hdf5',
                                                    'predict/scripts/*.joblib'], 
            },
)