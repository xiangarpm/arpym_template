
Python guidelines
================================

Coding style
--------------------------------

PEP 8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Always follow the `PEP 8 style guide <https://www.python.org/dev/peps/pep-0008>`_.
For naming rules, I suggest the following:

1. Functions, variables and attributes should be in ``lowercase`` or ``lowercase_with_underscores``. We shall not use more than three underscores (i.e. four words) unless in some special cases.

2. Package and module (script) names should be short, all ``lowercase``. In some special cases we may use one underscore.

3. Classes and exceptions should be in ``CapitalizedWords``.

4. Module-level constants should be in ``ALL_CAPS``.

For inline comments, please see `here <https://www.python.org/dev/peps/pep-0008/#comments>`_.
Note that we shall not use ``##`` to separate the codes like ``%%`` in the
Matlab. For docstrings (comments on modules, functions and classes), please see
section :ref:`docstrings_documentation`.

Below are the tools that we shall use to check the coding style automatically.

Spyder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Go to ``Preferences -> Editor -> Code Introspection/Analysis``, activate
``Real-time code style analysis``. You will see the yellow warning signs on the
left of the editor.

PyLint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use PyLint to check each script. Here is the
`tutorial <https://pylint.readthedocs.io/en/latest/tutorial.html>`_. In the
command line, simply run

::

    $ pylint script_name.py

And PyLint will list all the style problem and generate a report automatically,
as the following:

>>> ************* Module arpym_template.estimation.flexible_probabilities
>>> C:  1, 0: Missing module docstring (missing-docstring)
>>> C:  9, 8: Invalid attribute name "x" (invalid-name)
>>> C: 10, 8: Invalid attribute name "p" (invalid-name)
>>> C: 25, 8: Invalid variable name "x_" (invalid-name)
>>> C: 36, 8: Invalid variable name "t_" (invalid-name)
>>> E: 37,25: Module 'numpy' has no 'log' member (no-member)
>>> C: 40, 4: Invalid argument name "z" (invalid-name)
>>> C: 40, 4: Invalid argument name "h" (invalid-name)
>>> C: 55, 4: Invalid argument name "Type" (invalid-name)
>>> ...

.. _docstrings_documentation:

Docstrings and documentation
--------------------------------

We use `Sphnix <http://www.sphinx-doc.org/en/stable/index.html>`_ to generate
the documentation (in html, pdf or Latex) automatically. All the documentation,
including the Docstrings (comments on modules, functions and classes) should
use `reStructuredText <http://docutils.sourceforge.net/rst.html>`_ (reST)
syntax. Here is a simple `tutorial <http://www.sphinx-doc.org/en/stable/rest.html>`_.

For the Docstrings we shall always follow the
`Google Style <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google>`_.
Page :ref:`example_google` is the output of the module ``example-google.py`` of
Sphnix.

We can also add the icons with links in the Docstrings, for example:
|exercise| |code|

.. |exercise| image:: icon_ex_inline.png
    :scale: 20 %
    :target: https://www.arpm.co/lab/redirect.php?permalink=eb-rat-partition

.. |code| image:: icon-code-1.png
    :scale: 20 %
    :target: https://www.arpm.co/lab/redirect.php?codeplay=S_RatingPartitions



.. _guideline_variables:

Variables
--------------------------------

Arithmetic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure we always add the decimal point if we want to use float/double.

::

      1 / 3
      # SHOULD READ
      1.0 / 3.0

Although in Python3 it will give you the float number in the above operation, it
is always safe to use decimal point to prevent potential errors.

In Python we can assign multiple values on one line:

::

    i, j, k = 1, 2, 3
    x, y = 0.0



String
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Concatenate strings

>>> "a" + "b"
'ab'

Create a multiline string that contains ``"`` and ``'``

::

    s = """She said,
        "there's a good idea."
        """

Convert string to number

::

    x = int("2")
    y = float("2.5")


Boolean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For details, see `here <http://thomas-cokelaer.info/tutorials/python/boolean.html>`_.

In Python, the boolean variables are simply ``True`` and ``False``. The logic
operators are ``not``, ``and`` and ``or``. Also there are functions ``all()`` and
``any()`` just like Matlab.

>>> 1 == 1
True

>>> not 1 < 2
False

>>> True or 1 != 1
True

>>> all([True, False])
False

It is also easy to convert between boolean and integer.

>>> int(True)
1

>>> bool(1)
True

>>> int(False)
0

>>> bool(0)
False

But the logic operators can also apply to other variables! Every object has a
boolean value. The following elements are ``False``:

* ``None``
* ``0`` (whatever type from integer, float to complex)
* Empty collections: ``""``, ``()``, ``[]``, ``{}``
* Objects from classes that have the special method ``__nonzero__``
* Objects from classes that implements ``__len__`` to return ``False`` or zero

>>> not 'this is a string'
False

>>> not ''
True

>>> if 3.141592: print('This also is true!')
This also is true!

>>> if 0:
...     print('0 is true!')
... else:
...     print('0 is false!')
0 is false!

We should be very careful that ``and`` and ``or`` operators may have strange
outputs

>>> 'pi' and 3.141592
3.141592

>>> 'e' or 2.718281
'e'

So we should avoid the above cases in general. And we should be very
careful when we use the ``while`` loop. If you write ``while 10`` or
``while 'a'`` by mistake, it will become an infinity loop!


List, tuple and dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

List and dictionary are mutable and tuple is immutable, i.e. tuple is more
efficient in memory but you cannot change the values in tuple.

List and tuple are ordered and dictionary is unordered, i.e. you cannot access
a dictionary's elements using index 0, 1, 2, ...

In most cases we shall use tuple as the output of a function. If the output is
long and complicated, then we shall use dictionary.

::

    my_list = [1, 2]
    my_list[0] = 3

    my_tuple = (1,2)
    my_tuple[0] = 3  # Error!

    my_dict = {'first': 1, 'second': 2}
    my_dict[0] =3   # Error!
    my_dict['first'] = 3

In Python you can sort a list of tuples:

>>> tuple_list = [(3, 'C'), (3, 'B'), (3, 'A'),
                  (2, 'C'), (2, 'B'), (2, 'A'),
                  (1, 'C'), (1, 'B'), (1, 'A')]
>>> tuple_list.sort()
>>> tuple_list
    [(1, 'A'),
     (1, 'B'),
     (1, 'C'),
     (2, 'A'),
     (2, 'B'),
     (2, 'C'),
     (3, 'A'),
     (3, 'B'),
     (3, 'C')]

Conversion between list and tuple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> my_list = [1, 2, 3]
>>> my_tuple = (1, 2, 3)

Convert list to tuple

>>> tuple(my_list)
(1, 2, 3)

Convert tuple to list

>>> list(my_tuple)
[1, 2, 3]

Note that the use of ``[]``:

>>> tuple([my_list])
([1, 2, 3],)

>>> list([my_tuple])
[(1, 2, 3)]

Copy of a list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure you use ``b = a[:]`` instead of ``b = a`` when you want to make a copy
of a list ``a``.

``b = a`` means you have a new pointer ``b`` that points to the same address of
``a``. For example

>>> a = [1, 2, 3]
>>> b = a
>>> b[0] = 3
>>> a
[3, 2, 3]

This is also true for Numpy ``array`` and ``matrix``, which are indeed lists.

>>> a = np.array([1,2,3])
>>> b = a
>>> b[0]=3
>>> b
array([3, 2, 3])


List comprehensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use list comprehensions for *simple* ``for`` loops.
::

    a = []
    for x in range(10):
        if x % 2 == 0:
            a.append(x ** 2)
    # SHOULD READ
    a = [x**2 for x in range(0,10) if x % 2 == 0]

It is also easy to convert to Numpy ``array``
::

    a = np.array([x**2 for x in range(0,10) if x % 2 == 0])


Slicing the list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do not supply 0 for the ``start`` index or the length of the sequence for the ``end`` index, i.e.
::

    a[0:10]
    # SHOULD READ
    a[:10]

and
::

    a[5:len(a)]
    # SHOULD READ
    a[5:]

This rule also applies to numpy ``array`` and ``matrix``.

**Important: in Python you can use indices larger than the length of a list.**
::

    my_list = [1, 2, 3]
    my_list[:10]
    # this is equivalent to:
    my_list[:3]


Control flow
--------------------------------

List, range, enumerate, zip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We should avoid using the combination of ``range()`` and ``len()``. For example
::

    my_list = [5, 2, 9]

    for i in range(0, len(my_list)):
        print(2*my_list[i]+1)
    # SHOULD READ
    for item in my_list:
        print(2*item+1)

If we also want to access the indices, we can use the ``enumerate``. For example
::

    my_list = [5, 2, 9]

    for i, item in enumerate(my_list):
        # they are equal
        print(2*my_list[i]+1)
        print(2*item+1)

Sometimes it is also good to use ``zip`` function
::

    my_list1 = [5, 2, 9]
    my_list2 = [3, 8, 7]

    for item1, item2 in zip(my_list1, my_list2):
        print(item1 + item2)  # 8, 10, 16

``zip`` is more efficient in Python3 than in Python2, so it is good to use it in
a for loop if necessary.



Else block
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python provides a strange feature that allows you to use ``else`` after ``for``
and ``while``.
::

    for i in range(3):
        print("Loop %d" % i)
    else:
        print("Else block!")

>>>
Loop 0
Loop 1
Loop 2
Else block!

This feature looks very confusing. So we should **never** use it. But we should
aware this when we use some external Python codes.


Switch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unfortunately Python does not provide switch loop. You can solve this either
using multiple ``if``-``elif`` blocks or the dictionary as follows:

::

    def case1():
        return "this is case 1"


    def case2():
        return "this is case 2"


    def case3():
        return "this is case 3"


    def case4():
        return "this is case 4"

    options = {0: case1,
               1: case1,
               2: case2,
               3: case3,
               4: case4,
               5: case4}

>>> options[0]()
"this is case 1"


Functions
--------------------------------

Input and output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A typical Python function is defined as follows:
::

    def my_fun(input1, input2):
        # algorithm...
        return output1, output2

The output of ``my_fun`` is a tuple.
::

    output = my_fun(input1, input2)
    type(output)  # tuple (output1, output2)

Or we an simply write
::

    output1, output2 = my_fun(input1, input2)

When the output of a function has a complicated structure, it is better to use a
dictionary or class, see :ref:`guideline_variables`.


Default values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In Python we can define the default values of a function easily:

::

    def f(x, y = 10):
        return x + y

*Important:* only assign simple number, string, boolean or ``None`` as default
values, do not use "dynamic" values, such as ``list``. For example

::

    def f(a, L=[]):
        L.append(a)
        return L

    print(f(1))  # [1]
    print(f(2))  # [1, 2]
    print(f(3))  # [1, 2, 3]

To solve the problem above, you can always use ``None`` instead.

::

    def f(a, L=None):
        if L is None: L = []
        L.append(a)
        return L

    print(f(1))  # [1]
    print(f(2))  # [2]
    print(f(3))  # [3]

This also holds for Numpy ``array`` and Pandas ``DataFrame`` which are indeed
lists.

Python function also has default return values, which is ``None``.

>>> def f(): print("No return!")
>>> f() is None
No return!
True


Arbitrary arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Python you can set arbitrary number of inputs, using ``*args``

::

    def f(x, *args):
        for i in args:
            x = x + i
        return x

    f(1, 2)  # 3
    f(1, 2, 3)  # 6
    f(1, 2, 3, 4)  # 10

::

or ``**kwargs`` for keyword arguments

::

    def f(x, *kwargs):
        for key in kwargs:
            print(key)
            x = x + kwargs[key]
        return x
::

>>> f(1, second=2, third=3)
second
third
6

But sometimes this can cause unexpected errors. So we shall not this except for
some special cases.


Keyword and positional arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the function has a lot of inputs, for example

::

    def f(x, y, z, option_1=1, option_2='a', option_3=True):
        pass

Then something like ``f(1, 2, 3, option_1=2, 'a')`` will cause errors. We want
to "force" users to clarify ``option_1``, ``option_2``, ``option_3`` everytime
they use the function to avoid potential bugs. To do this we can define:

::

    def f(x, y, z, *, option_1=1, option_2='a', option_3=True):
        print('Good!')

>>> f(1, 2, 3, 2, 'b', False)
TypeError: f() takes 3 positional arguments but 6 were given
>>> f(1, 2, 3, option_1=2, option_2='b', option_3=False)
'Good!''


Function is an object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In Python, functions is an object, i.e. it can be treated as a variable. For
example:

>>> def my_add(x, y): return x + y
>>> f = my_add
>>> type(f)
function
>>> f(3,2)
5

>>> def foo(f, x, y): return f(x,y)
>>> foo(my_add, 1, 2)
3

>>> def my_minus(x, y): return x - y
>>> f_list = [my_add, my_minus]
>>> f_list[1](2,1)
1


Function scope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The variables defined in a script are all global. You can always access global
variables in a function.

::

    c = 10  # c is a global variable

    def f(x):
        return c*x

    f(1)  # 10

However functions are not able to modify the value of the global variables.

::

    c = 10  # c is a global variable

    def f(x):
        c = 5  # here c is a local variable
        return c*x

    f(1)  # 5
    c  # still 10

We can use the ``global`` command to solve this problem.

c = 10  # c is a global variable

def f(x):
    global c  # now the function has the right to change c
    c = 5  # here c is a local variable
    return c*x

f(1)  # 5
c  # 5

The same problem applies to

::

    def f(x):
        c = 10  # local variable in f
        def g(y):
            return c*y  # g can assess all the variables in f
        return g(x)

    f(1)  # 10

With ``global`` you can also change ``c`` in ``g``. But in that case you can
even access ``c`` out of ``f``.

::

    def f(x):
        c = 10  # local variable in f
        def g(y):
            global c
            c = 5
            return c*y
        return g(x)

    f(1)  # 5
    c  # 5

Then instead of ``global`` we should use ``nonlocal``.

::

    def f(x):
        c = 10  # local variable in f
        def g(y):
            nonlocal c
            c = 5
            return c*y
        return g(x), c

    f(1)  # (5, 5)
    c  # NameError: name 'c' is not defined




Lambda functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In Python, we can define anonymous functions similar to ``@(x)...`` functions in
Matlab.

>>> f = lambda x: x ** 2
>>> f(2)
>>> 4


Classes
--------------------------------

To illustrate why we need classes, consider the following pseudo code for ARMA
model.
::

      # Estimating ARMA parameters
      ar, ma, sigma2 = arma_fit(data, p, q, method, options)

      # Extracting invariants (residuals)
      x = arma_res(data, ar, ma, sigma2)

      # Generating Monte Carlo scenarios of an ARMA process
      y = arma_simulate(y_tnow, x_tnow, j_bar, t_hor, ar, ma, sigma2)

      # Predict multi period expected values of ARMA
      y_hat = arma_predict(y_tnow, x_tnow, t_hor, ar, ma, sigma2)

The above code looks complicated and the functions have a lot of arguments. This
can be simplified by creating a class called ``ARMA``.

::

      # Initialize the class
      model = ARMA(p, q)

      # Estimating ARMA parameters
      model.fit(data, method, options)

      # Extracting invariants (residuals)
      x = model.res(data)

      # Generating Monte Carlo scenarios of an ARMA process
      y = model.simulate(j_bar, t_hor)

      # Predict multi period expected values of ARMA
      y_hat = model.predict(t_hor)

A class can be defined as follows:

::

      class ARMA(object):

          # Constructor
          def __init__(self, p, q):
              # these variables are called attributes
              self.p = p
              self.q = q
              self.ar = np.zeros(p)
              self.ma = np.zeros(q)
              self.sigma2 = 1.0

          # The following functions are called methods
          def fit(self, data, method, options):
              # update self.ar, self.ma and self.sigma2
              # also create attributes self.y_tnow, self.x_tnow from the data
              return 0

          def res(self, data):
              return 0

          def simulate(self, j_bar=1000, t_hor=1):
              return 0

          def predict(self, t_hor=1):
              return 0

For practical examples, please see :ref:`flexible_probabilities`. Also see
https://docs.python.org/3/tutorial/classes.html.





Modules
--------------------------------

Beginning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The beginning of each module should have the following format
::

    # -*- coding: utf-8 -*-
    ```General short introduction, with refs to the Lab if necessary
    ```
    import modules

From `PEP 8 <https://www.python.org/dev/peps/pep-0008>`_:

*Imports should be grouped in the following order:*

1. *standard library imports (``os``, ``math``, ``sys``)*
2. *related third party imports (``numpy``, ``pandas``, ``scipy``, ``sklearn``)*
3. *local application/library specific imports*

*You should put a blank line between each group of imports.*

We shall use ``np`` and ``pd`` as the acronyms for ``numpy`` and ``pandas``
respectively. Let us discuss if you think we should use other acronyms.


Clear
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In Python, the function ``clear()`` plays the same role as the Matlab command
``clc``.

If you want to remove a variable from the enviorment, you can use ``del()``.

>>> a = 1
>>> del(a)
>>> a
NameError: name 'a' is not defined

However there is no convenient way to remove all the variables. Because Python
enviorment always includes some default variables. So we do not recommend to do
something like ``clear all; close all; clc;`` at the begining of Python scripts.



Numpy
--------------------------------

For details, see `NumPy for Matlab users <https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html>`_

Array and matrix are different
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Matlab, array or vector is a matrix. In Numpy, ``matrix`` is a subclass of
``array``.

>>> M = np.matrix([[1,2,3],[4,5,6]])
>>> A = np.array([[1,2,3],[4,5,6]])
>>> type(M)
>>> numpy.matrixlib.defmatrix.matrix
>>> type(A)
>>> numpy.ndarray

**Since ``array`` is the default of Numpy, we should prefer it to ``matrix``, even
the later is designed to be Matlab-like. We should use ``matrix`` only when it
is necessary.**

The key differences are as follows:

1. Operator ``*``, ``dot()``, and ``multiply()``:

    * For ``array``, ‘``*``’ means element-wise multiplication, and the ``dot()``
      function is used for matrix multiplication.
    * For ``matrix``, ‘``*``’ means matrix multiplication, and the ``multiply()``
      function is used for element-wise multiplication.

2. Handling of vectors (one-dimensional arrays)

    * For ``array``, the vector shapes 1xN, Nx1, and N are all different things.
      Operations like ``A[:,1]`` return a one-dimensional array of shape N, not a
      two-dimensional array of shape Nx1. Transpose on a one-dimensional array
      does nothing.
    * For ``matrix``, one-dimensional arrays are always upconverted to 1xN or Nx1
      matrices (row or column vectors). ``A[:,1]`` returns a two-dimensional
      matrix of shape Nx1.

3. Handling of higher-dimensional arrays (ndim > 2)

    * ``array`` objects can have number of dimensions > 2;
    * ``matrix`` objects always have exactly two dimensions.

4. Convenience attributes

    * ``array`` has ``a .T`` attribute, which returns the transpose of the data.
    * ``matrix`` also has ``.H``, ``.I``, and ``.A`` attributes, which return
      the conjugate transpose, inverse, and ``asarray()`` of the matrix,
      respectively.

5. Convenience constructor
    * The ``array`` constructor takes (nested) Python sequences as initializers.
      As in, ``array([[1,2,3],[4,5,6]])``.
    * The ``matrix`` constructor additionally takes a convenient string
      initializer. As in ``matrix("[1 2 3; 4 5 6]")``.


Pandas
--------------------------------
We can use Pandas to import .csv data easily.

::

    import os

    import pandas as pd

    # Get the path of the current folder (where your .py file located. )
    path = os.path.dirname(os.path.abspath(__file__))

    # Suppose that the data is placed in the same folder
    filename = path + 'my_data.csv'

    df = pd.read_csv(filename, header=None)



Debug
--------------------------------
