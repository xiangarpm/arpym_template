from setuptools import setup
# from distutils.core import setup


setup(
    name='arpym_template',
    version='0.1',
    description='A simple example of a Python package',
    url='https://www.arpm.co',
    # long_description=readme,
    author='ARPM',
    author_email='info@arpm.co',
    packages=['arpym_template', 
              'arpym_template.estimation',
              'arpym_template.tests'],
    classifiers=[
        'Development Status :: 3 - Alpha'
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=['numpy','pandas'],
    python_requires='>=3',
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False
)
