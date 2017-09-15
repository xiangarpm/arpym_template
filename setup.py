from distutils.core import setup

with open('README.rst') as f:
    readme = f.read()

setup(
    name='arpym_template',
    version='0.1',
    description='A simple example of a Python package',
    long_description=readme,
    author='ARPM',
    author_email='xiang.shi@arpm.co',
    packages=['arpym_template', 'arpym_template.tests'],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ]
)
