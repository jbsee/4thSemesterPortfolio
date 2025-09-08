---
title: Guide To Python
publish: true
---

> [!warning] ***You don't need to learn Python. You need to use it. Let frustration be your teacher.***
>
>This guide was built using Chat-GPT and Gemini. It started as a few questions about Python and wasn't meant to be this comprehensive, but at some point I decided to treat this as a LLM-learning experiment: **Can I use LLMs to help generate useful, customized learning material for a programming language I barely know?**
>
>The biggest risk seems to be that I don't know what I don't know—I can't really tell if anything is missing. I tried mitigating this risk by having Chat-GPT's reasoning model fight Gemini's reasoning model in critiquing and correcting the guide. After a while I arrived at a point where corrections and additions seemed somewhat trivial, both models seemed happy about what was included, so I settled on a version of the guide. I have reviewed everything in the guide and modified quite a bit, but there are probably still blind spots and errors I've missed. The guide will be updated continuously throughout the semester—unless I abandon it. It might turn out to be a waste of time. Or a gold mine. Most likely, it's somewhere in between. 
> 
> **This resource might be inaccurate and is not in any way a complete guide. Double check anything important!**

This is meant as an introduction to Python—not to be memorized, but to help get a general understanding of the language. Some things might not make a lot of sense during the first read-through—and some (e.g. comprehensions) are not strictly necessary to implement straight away—however, they are nice to know about. Use the guide for look-ups while learning by doing.

Some comparisons to C# are included, since that's the language I'm most comfortable with.

The following resources might also be helpful:
- https://learnxinyminutes.com/python/
- https://gto76.github.io/python-cheatsheet/

## The Big Picture: Philosophy & Ecosystem

> [!info]- **From C# to Python: Mental Model Shift – From Architect to Sculptor**
>
> In **C#**, code is shaped by static typing and explicit contracts.  
> - You define interfaces and classes up front because the compiler needs to know exact shapes at compile time.  
> - Errors are caught early because type checking, overload resolution, and nullability rules are enforced before the program runs.  
> - This enables tooling (e.g. IntelliSense, refactoring) to operate with confidence.
>
> In **Python**, code is shaped by dynamic typing and runtime behavior.  
> - You pass objects based on the methods or attributes they support (duck typing), not based on declared types.  
> - Type mismatches and missing methods raise exceptions *at runtime*, not before.  
> - This reduces boilerplate and accelerates development, but shifts correctness checks to testing.
>
> **Trade-off**:  
> - C# gives you safety guarantees, but requires more upfront structure.  
> - Python gives you faster iteration and simpler syntax, but relies on discipline and tests to catch issues.
>
> **Or in other words:**  
> - C#: Define what *should* happen, let the compiler enforce it.  
> - Python: Try what *might* work, let the runtime crash if it doesn’t.
  
#### **1. The Zen of Python**  
`import this`. Read it. Then break half of it with a good reason.

> [!info]- **import this**
> 1. Open your **terminal**.
> 2. Type `python` or `python3` and hit Enter.
> 3. `import this`
> 4. Congrats. You’ve joined a cult.

#### **2. The ecosystem is fractured** 
Python isn’t one neat thing—it’s a tangle of versions, tools, and packaging formats. Consistency is survival.

> [!info]- **Quick ecosystem map**
>
> #### Versions
> - Python 2.x: Dead but haunts legacy systems.
> - Python 3.x: Actively developed, but 3.6 → 3.12+ changes aren't trivial.
> 
> #### Environments
> - `venv`: Built-in. Creates isolated environments. Use it.  
> - `conda`: Cross-platform, includes packages + Python version.
> - `system Python`: Just don’t. Too easy to nuke your OS tools.  
> - Docker: For reproducibility, deployment, and containment.
> 
> #### Package managers
> - `pip`: The default. Works well, especially with `venv`. `pip freeze` **is not a real lockfile**—it just lists everything currently installed, including transitive deps. Use `pip-tools compile` or `poetry.lock` for reproducible builds. 
> - `conda`: Comes with Anaconda. Great for data science—includes Python itself and native libs—but bloated outside that niche.  
> - `poetry`, `pipenv`: Modern dependency managers with real lockfiles. Poetry is opinionated and rising fast; Pipenv is fading.
> - `pipx`: Installs **single CLI apps** in isolated venvs—no project pollution. Perfect for tools like `black`, `ruff`, or `pytest`.
>
> #### Project setup
> - `pyproject.toml`: Centralized config for build systems and dependencies.
> - Supported by `poetry`, `flit`, and increasingly by `pip`.
> - Slowly replacing `setup.py`, `requirements.txt`, and friends.
>
> **But!** You need to pick exactly **one** build backend:
> - `setuptools`, `hatchling`, `flit`, or `poetry-core`.
> - Declare it explicitly in `pyproject.toml`:
>   ```toml
>   [build-system]
>   requires = ["setuptools>=68", "wheel"]
>   build-backend = "setuptools.build_meta"
>   ```
> - Mixing backends = `python -m build` rage spirals and unhelpful errors.
> ---
>
> #### Bottom line:
> Pick one stack early:  
> _e.g., `venv` + `pip` + `pyproject.toml` + `ruff/black/mypy`_.
> Then **stick to it**. Mixing tools leads to madness.

> [!info]- **Further reading**
>
> Once you’re building real packages, look into `setuptools`, `build`, and `twine`. Publishing to PyPI is its own world—but totally separate from learning the language itself.

#### **3. It’s slow**  
Python isn’t built for speed. It’s built for flexibility. You pay for that in raw performance.

> [!info]- **What slows it down**
>
> - **GIL** (Global Interpreter Lock): Only one thread runs Python bytecode at a time.
> - **Interpreted bytecode**: Code is compiled to bytecode, then interpreted.
> - **Dynamic typing**: Every operation involves type lookups and checks.
> - **Boxing/unboxing**: Even simple math gets wrapped in objects.
>
> #### Example:
> ```python
> import time
>
> def slow():
>     total = 0
>     for i in range(10_000_000):
>         total += i
>     return total
>
> start = time.time()
> slow()
> print(time.time() - start)
> ```
>
> Try that in C, Rust, or even Java—Python will lose every time.
>
> ---
>
> #### But!
> - You can offload to C via extensions (`numpy`, `cython`, etc.).
> - You can parallelize with `multiprocessing` (CPU-bound) or use `asyncio` / threads (IO-bound).
> - Async only works **if your coroutines `await` properly**—any blocking call still halts everything.
> - Yes, it’s slow. But clear code beats premature optimization. Profile first, then patch the hotspots when you run into performance issues.
>
> ---
>
> **Use Python for speed of _development_, not execution.**

## The Practical Workflow: Setup, Tooling & Testing

#### **4. Virtual Environments Are Non-Negotiable**  
Unlike .NET, Python has no per-project dependency isolation by default. Use `venv`, or suffer.

> [!info]- **The problem**
>
> - `pip install` installs globally or per-user.  
> - Conflicting project dependencies = chaos.
>
> ---
>
> #### The fix: venv
>
> ```bash
> python -m venv .venv           # Create env
> source .venv/bin/activate      # Linux/macOS
> .venv\Scripts\activate         # Windows
> pip install -r requirements.txt
> ```
>
> - Keep a `.venv` folder per project.
> - Add it to `.gitignore`.
>
> Like `node_modules`, but saner.

#### **5. Configuration and Environment Variables**  
C# has `appsettings.json`, `IConfiguration`, `UserSecrets`, etc.  
Python gives you `os.getenv()` and leaves the rest up to you. There’s no official standard—just conventions.

> [!info]- **How Python handles config**
>
> #### Environment variables
> ```python
> import os
>
> db_url = os.getenv("DATABASE_URL", "sqlite:///default.db")
> debug = os.getenv("DEBUG", "false").lower() == "true"
> ```
> - Always returns strings—cast manually if needed  
> - Use for secrets, flags, DB URLs, etc.  
> - Safe fallback: provide default values  
>
> #### `.env` files (local dev)
> ```
> DEBUG=true
> DATABASE_URL=postgres://localhost
> SECRET_KEY=supersecret
> ```
> - Not part of the standard library  
> - Use `python-dotenv` to load them  
>
> #### Install:
> ```bash
> pip install python-dotenv
> ```
> Then:
> ```python
> from dotenv import load_dotenv
> load_dotenv()  # loads .env into os.environ
> ```
>
> #### Wrap it:
> ```python
> class Config:
>     DB = os.getenv("DATABASE_URL")
>     DEBUG = os.getenv("DEBUG", "false").lower() == "true"
> ```
> Keeps config centralized and testable.
>
> #### C# comparison:
> - `appsettings.json` → `.env` file  
> - `IConfiguration` → your own wrapper on `os.getenv()`  
> - `UserSecrets` → external secret stores or `.env`, depending on infra  
>
> Python doesn’t enforce structure. Most devs roll their own or use something like `dynaconf`, `pydantic`, or `environs`.

#### **6. Going Beyond Scripts: Project Layouts & Packaging**

> [!info]- **From Scripts to Projects**
> Once your `.py` files start to breed, you’ll want structure. Python doesn’t enforce it, but **modern best practices** do:
>
> ```text
> my_project/
> ├── .venv/                 # Local virtual environment
> ├── src/                   # Your actual code lives here
> │   └── my_package/
> │       ├── __init__.py
> │       └── main.py
> ├── tests/                 # Unit tests
> │   └── test_main.py
> ├── pyproject.toml         # Build config and dependencies
> └── README.md              # You won't update this
> ```
>
> - **Why `src/`?** Avoids import bugs during testing (accidentally importing from current dir instead of installed package).
> - **What’s `pyproject.toml`?** Like a `.csproj` or `package.json`. Defines your package and dependencies.
> - **Want to test your code live?**
>   ```bash
>   pip install -e .
>   ```
>   This does an *editable install*—your package is importable even as you edit the files.
> - **Build a `.whl` file?**
>   ```bash
>   python -m build
>   ```
>   This creates a **wheel**—Python’s version of a NuGet `.nupkg` file.

#### **7. Testing: The Pythonic Way**

> [!info]- **Why `pytest` Feels Like Cheating**
> Python testing is **minimalist, expressive, powerful**. No base classes. No ceremony. Just functions.
>
> ```python
> # test_math.py
> def test_add():
>     assert 2 + 2 == 4
> ```
>
> - **No inheritance, no attributes, no boilerplate.**  
> - `assert` is native—it works because `pytest` rewrites the AST to give detailed failures. 
> - Failures tell you **exactly** what broke:
>   ```
>   assert 2 + 2 == 5
>          +   +   -
>          4       5
>   ```
>
> ---

> [!info]- **Fixtures = Dependency Injection**
 > Replaces `SetUp`/`TearDown` with something smarter:
 >
 > ```python
 > import pytest
 >
 > @pytest.fixture
 > def db():
 >     conn = connect_to_db()
 >     yield conn
 >     conn.close()
 >
 > def test_query(db):
 >     result = db.query("SELECT * FROM users")
 > ```
 >
 > Fixtures are composable, reusable, and scoped.  
 > You can even **parametrize** them to test variants.

#### **8. Python is opinionated about formatting—PEP 8, black, etc**. 
The language doesn’t enforce formatting, but the ecosystem does.  
Use a formatter like `black` and move on. Don’t waste time on style debates—just let the robot win.

> [!info]- **PEP 8, linters, and formatters**
>
> #### PEP 8:
> - Official Python style guide.
> - Covers naming, spacing, indentation, imports, line length, etc.  
> - Used by most tools (`ruff`, `black`, `isort`).
>
> #### `black`:
> - Opinionated auto-formatter.
> - Ignores most config—just formats consistently.
> ```bash
> black my_script.py
> ```
> - No decisions, no arguments. That’s the point.
>
> #### `ruff`:
> - Linter **and** fixer, written in Rust.
> - 10–50× faster than `flake8` or `pylint`.
> - Bundles most `flake8` plugins + autofix on save.
> - Can **replace** both `flake8` and `pylint` in one go.
> ```bash
> pipx install ruff      # or: pip install ruff
> ruff check .
> ```
> - Also runs `black`-style formatting (`ruff format .`)—one tool, all roles.
>
> #### VS Code:
> - `Ctrl+Shift+I` runs **Format Document**.
> - It only uses `black` (or `ruff`) if you’ve set it:
>   ```json
>   "python.formatting.provider": "black",
>   "[python]": {
>     "editor.formatOnSave": true
>   }
>   ```
> - Otherwise, it might run `autopep8`, `yapf`, or nothing at all.
>
> #### `flake8` / `pylint`:
> - Static analyzers.
> - Check for style **and** potential bugs.
> - Can be run in CI or hooked into editors.
>
> ---
>
> #### Example:
> Input:
> ```python
> def  f(x):return(x+1)
> ```
> After `black`:
> ```python
> def f(x):
>     return x + 1
> ```
>
> ---
>
> If you're writing solo scripts, do whatever.  
> In shared code? Standardize and automate it.

#### **9. Docstrings and the `help()` function**  
Python's docstrings are live, introspectable, and baked into the runtime. They're not just comments—they're metadata.

> [!info]- **What docstrings are**
>
> A docstring is a string literal placed as the first statement in a module, function, class, or method.
>
> ```python
> def add(x, y):
>     """Adds two numbers together."""
>     return x + y
>
> print(add.__doc__)
> help(add)
> ```
>
> - `help()` displays the docstring in a readable format.
> - Tools like `pydoc` or IDEs hook into this system.
> - Multi-line docstrings use triple quotes.
>   
> Python expects you to document—not for style, but because your code lives in a REPL.

## Core Idioms: The Pythonic Way
Syntax and patterns that are considered "Pythonic"—the natural way to write code Python.

#### **10. Indentation _is_ the syntax**  
Blocks are defined by consistent spacing, not `{}`. Misalign and it breaks. This forces readable code, but kills copy-paste from sloppy sources. Tabs vs spaces matters—pick one (spaces) and stick to it.

#### **11. Comprehensions are idiomatic**  
List, dict, set comprehensions—get used to these. They’re faster, cleaner, and Pythonic™. Same with generator expressions and `zip`, `enumerate`, etc.

> [!info]- **Cheat Sheet: Comprehensions & Friends**
> 
> **Clunky, unidiomatic**
> ```python
> squares = []
> for x in range(10):
>     squares.append(x ** 2)
> ```
> 
> **Idiomatic**
> ```python
> squares = [x**2 for x in range(10)]
> ```
> 
> Same result. But the second version is **natural to the language**.
> 
> #### List comprehension
> ```python
> squares = [x**2 for x in range(10)]
> # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
> ```
> 
> #### Conditional list comprehension
> ```python
> evens = [x for x in range(10) if x % 2 == 0]
> # [0, 2, 4, 6, 8]
> ```
> 
> #### Dict comprehension
> ```python
> squares = {x: x**2 for x in range(5)}
> # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
> ```
> 
> #### Set comprehension
> ```python
> uniques = {x % 3 for x in range(10)}
> # {0, 1, 2}
> ```
> 
> #### Generator expression (lazy)
> ```python
> lazy_squares = (x**2 for x in range(10))
> next(lazy_squares)  # 0
> ```
> 
> #### Nested list comprehension
> ```python
> grid = [(x, y) for x in range(3) for y in range(2)]
> # [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
> ```
> 
> ---
> 
> #### Bonus: `zip` + `enumerate`
> ```python
> names = ['a', 'b', 'c']
> scores = [10, 20, 30]
> 
> paired = list(zip(names, scores))
> # [('a', 10), ('b', 20), ('c', 30)]
> 
> for i, name in enumerate(names, start=1):
>     print(i, name)
> # 1 a
> # 2 b
> # 3 c
> ```
> 
> ---
> 
> **C# comparison**  
> Comprehensions often fill the same role as LINQ’s `Select`, `Where`, and `ToList`:  
> expressive, one-liner transformations and filters over collections.  
> Think `[x for x in data if x > 0]` as Python’s answer to `.Where(x => x > 0).ToList()`.
> 
> **Rule of thumb**  
> If you're writing a `for` loop just to build a list, you're probably missing a comprehension.
#### **12. Slicing is a superpower**  
Python’s slicing syntax is more than just `[:]`. It’s a full-on sequence manipulation toolkit. Lists, strings, tuples—all obey the same `[start:stop:step]` format. Negative indices count from the end. The `step` lets you skip, stride, or reverse.

> [!info]- **Slicing syntax and power**
>
> Basic form:
>
> ```python
> seq[start:stop:step]
> ```
>
> - `start`: where to begin (inclusive)  
> - `stop`: where to end (exclusive)  
> - `step`: how much to move each time
>
> ---
>
> #### Examples:
>
> ```python
> numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>
> numbers[2:7]     # [2, 3, 4, 5, 6]      elements 2 through 6
> numbers[:5]      # [0, 1, 2, 3, 4]      start omitted = 0
> numbers[5:]      # [5, 6, 7, 8, 9]      end omitted = to the end
> numbers[::2]     # [0, 2, 4, 6, 8]      every second element
> numbers[::-1]    # [9, 8, 7, ..., 0]    reversed
> numbers[-3:]     # [7, 8, 9]            last 3 elements
> numbers[:-3]     # [0, 1, ..., 6]       everything but the last 3
> ```
>
> ---
>
> #### Why it matters:
>
> - Works on **any** sequence: strings, lists, tuples  
> - Unified syntax replaces clunky loops  
> - Elegant way to express reversals, sublists, and patterns
>
> ---
>
> #### C# comparison:
>
> In C# you might use:
>
> ```csharp
> var slice = list.Skip(2).Take(5).ToList();
> var reversed = list.Reverse().ToList();
> ```
>
> In Python, you just write:
>
> ```python
> slice = list[2:7]
> reversed = list[::-1]
> ```
>
> ---
>
> Slicing is expressive, fast, and core to idiomatic Python. If you're using loops where slices would do—you're writing Java in a trench coat.

#### **13. F-strings are king**  
Forget ̣̣̣`$`, .format()` and `%` formatting. F-strings (`f"Hello, {name}!"`) are faster, cleaner, and support expression eval. Use them. Abuse them. Worship them.

> [!info]- **F-strings**
> F-strings are Python's string interpolation on steroids.
>
> #### Basic usage:
> ```python
> name = "Preben"
> print(f"Hello, {name}!")  # Hello, Preben!
> ```
>
> #### Expressions work too:
> ```python
> x = 3
> print(f"2x = {2 * x}")  # 2x = 6
> ```
>
> #### Format specifiers:
> ```python
> pi = 3.14159
> print(f"{pi:.2f}")  # 3.14
> print(f"{42:08}")   # 00000042
> ```
>
> #### Debug syntax (Python 3.8+):
> ```python
> val = 99
> print(f"{val=}")  # val=99
> ```
>
> ---
>
> #### Don’ts:
> - Don't concatenate strings manually (`"Hi " + name`)
> - Don't use `.format()` unless you’re stuck on Python 2
> - Don't try to use `%` formatting unless you're nostalgic for C

#### **14. The underscore has _roles_**  
Underscores aren’t just style—they signal intent, scope, or behavior.  
Some are enforced by Python. Others are cultural warnings. All mean something.

> [!info]- **Underscore patterns and examples**
>
> **`_` — last result in the REPL**  
> Python’s interactive shell stores the last result in `_`.
> ```python
> >>> 2 + 3
> 5
> >>> _ * 2
> 10
> ```
>
> ---
>
> **`_var` — soft "private"**  
> A naming convention for internal use. Not enforced.
> ```python
> def _helper(): ...
> obj._internal_data  # not for public use
> ```
>
> ---
>
> **`__var` — name mangling**  
> Used inside classes to avoid accidental name clashes.
> ```python
> class A:
>     def __init__(self):
>         self.__secret = 42
>
> A().__dict__  # {'_A__secret': 42}
> ```
>
> ---
>
> **`__var__` — dunder methods**  
> Hooks into Python's internals. Don’t make your own unless you're extending the language.
> ```python
> class User:
>     def __str__(self):
>         return "User instance"
>
> print(User())  # triggers __str__
> ```
>
> ---
>
> **`var_` — avoids keyword clash**  
> Used when a name would conflict with a keyword.
> ```python
> class_ = "Physics"
> ```
>
> ---
>
> **`for _ in ...` — discard value**  
> Common in loops when you don’t care about the variable.
> ```python
> for _ in range(3):
>     print("repeat")
> ```
>
> ---
>
> Each form signals something. Python may not enforce it, but readers (and linters) will.

#### **15. `__name__ == "__main__"` isn’t just boilerplate**  
It's how scripts decide whether to run or be imported. Core to modular design, testing, and avoiding side effects.

> [!info]- **How and why it works**
>
> Every Python file is a module. When run directly, Python sets:
>
> ```python
> __name__ == "__main__"
> ```
>
> But if it’s imported from another module:
>
> ```python
> __name__ == "module_name"
> ```
>
> ---
>
> #### Why it matters
>
> ```python
> def main():
>     print("Running script")
>
> if __name__ == "__main__":
>     main()
> ```
>
> This means:
> - `main()` runs when you execute the file directly.
> - It does **not** run when the module is imported.
>
> ---
>
> #### Good habits
>
> - Wrap script logic under `if __name__ == "__main__":`
> - Keeps modules **import-safe.**
> - Makes testing easier.
> - Separates **definitions** from **execution.**

#### **16. Errors are meant to happen**  
"**Ask forgiveness, not permission**" is core Python dogma. Instead of checking if a file exists, try opening it and catch the error. Learn to use `try/except` well.

> [!info]- **Example: try/except vs if-checking**
> 
> #### Unidiomatic (permission-checking):
> ```python
> if os.path.exists("data.txt"):
>     with open("data.txt") as f:
>         data = f.read()
> ```
> This **looks safe**, but it’s not. The file could be deleted between the check and the open. Classic race condition. The `os.path.exists()` tells you the file exists _at that moment_, but the world keeps spinning. Another process could delete or move it right after, and `open()` would still crash.
> 
> Don’t ask permission—**just try and handle failure**. It’s safer, simpler, and aligns with Python’s “Easier to ask forgiveness than permission” (EAFP) philosophy.
> 
> ---
> 
> #### Idiomatic (forgiveness-based):
> ```python
> try:
>     with open("data.txt") as f:
>         data = f.read()
> except FileNotFoundError:
>     data = None
> ```
> Now it’s race-proof. You **assume** it’ll work and handle failure explicitly. This is Python's philosophy in practice.
> 
> ---
> 
> #### More general form:
> ```python
> try:
>     risky_thing()
> except (ValueError, TypeError) as e:
>     handle(e)
> else:
>     do_when_no_exception()
> finally:
>     always_cleanup()
> ```
> - `except` catches errors  
> - `else` runs only if no error occurred  
> - `finally` always runs—error or not

> [!info]- **Common Built-in Exceptions**
>
> Python throws early and often. Idiomatic code doesn’t try to prevent all errors—it handles the ones that matter.
>
> #### Useful to catch:
>
> - `KeyError`:  
>   Dictionary key not found  
>   → Use `d.get("key", default)` if fallback makes sense
>    
>
> - `IndexError`:  
>   List/tuple index out of bounds  
>   → Use slicing (`x[:10]`) to avoid overreaching
>    
>
> - `AttributeError`:  
>   Missing method or attribute  
>   → Often a typo or misuse of duck typing
>    
>
> - `TypeError`:  
>   Operation on incompatible types (`"5" + 1`)  
>   → Python won’t auto-coerce like JS
>    
>
> ---
>
> Don’t catch `Exception` blindly. Trap what you expect. Let the rest crash and burn—with a traceback.

> [!info]- **Further reading**
>
> Testing tools like `unittest`, `pytest`, and `doctest` live outside the core language—but they're essential for serious projects. Learn them once you're writing more than scripts.


## Under the Hood: How Python Thinks About Data & Memory
Runtime, types, variables, and memory behind the scenes.

#### **17. The runtime is an interpreter**  
Python **compiles your code to bytecode**, then runs it on a **virtual machine**—a stack-based interpreter. This happens automatically, so it _feels_ like your code is read top to bottom, line by line. There's no manual compile step unless you bundle it with something like `pyinstaller`. Meaning: **everything is live**. You can REPL (Read–Eval–Print Loop) your way through anything.

> [!info]- **REPL vs Script**
> - **REPL** = screwdriver. Tighten one bolt. Test one idea.  
> - **Script** = blueprint. Run the whole thing.
>
> Both use the **same interpreter**, but the **workflow** is radically different from C#.  
> In C#, you write code, compile it, then run the binary. In Python, you just _run_ it. No build step. No type declarations. No ceremony.
> 
> #### REPL as core workflow
> Python encourages:
> - Writing and testing expressions live  
> - Prototyping ideas before saving them  
> - Treating the interpreter as a **scratchpad for thinking**
>
> In practice: many Python devs write entire programs one function at a time in a REPL, then paste it into a script once it works.
> It’s fast, messy, effective—and completely **idiomatic**: part of the ecosystem’s DNA.
>
> ---
>
> #### Launch REPL:
> ```bash
> python
> ```
> Or use `ipython` for extras like history, autocomplete, and color.
>
> ---
>
> #### Handy REPL tools:
> - `dir(obj)` — list attributes  
> - `type(obj)` — get type  
> - `help(obj)` — show docstring  
> - `id(obj)` — memory address  
> - `obj.__dict__` — internal state (if present)  
> - `dis(obj)` — disassemble to bytecode
>
> ---
>
> It’s not just a console—it’s **surgery without gloves**.  
> Poke live objects. Redefine functions mid-run.  
> Think of it as your debugger, test runner, and notepad—all in one.

#### **18. Compilation is real**  
Python _is_ compiled—to bytecode. That’s what the `.pyc` files in `__pycache__` are. It just happens behind the scenes, automatically, every time you run a script.

> [!info]- **What actually happens**
>
> #### Step-by-step:
> - You write code in a `.py` file.
> - The interpreter **compiles it to bytecode** (an intermediate format).
> - The bytecode is saved as `.pyc` in a `__pycache__` folder.
> - The VM (CPython) **executes** the bytecode line by line.
>
> This is why Python feels like an interpreted language—but it's technically **interpreting compiled bytecode**.
>
> ---
>
> #### View the bytecode:
> ```python
> import dis
>
> def square(x):
>     return x * x
>
> dis.dis(square)
> ```
> Output looks like this:
> ```
> 2           0 LOAD_FAST                0 (x)
>             2 LOAD_FAST                0 (x)
>             4 BINARY_MULTIPLY
>             6 RETURN_VALUE
> ```
> This is Python’s virtual instruction set—what the interpreter actually runs.
>
> ---
>
> #### So what?
> - Python doesn’t skip compilation—it just hides it.
> - You rarely need to care—unless you’re debugging performance or writing tooling.
> - Tools like `dis`, `compile()`, and `codeop` let you poke into the compiled layer.
>
> ---
>
> Leave this stuff alone unless:
> - You’re profiling hot code paths.
> - You’re building a language tool, debugger, or tracer.
> - Or you're just curious what Python's actually doing under the hood.

#### **19. All values are objects**  
Functions, types, even `None`—everything is an instance of something. Python runs on objects, not primitives. This makes metaprogramming a breeze but can be slippery. Don’t need to fully internalize it up front—just know it’s _not_ just syntactic sugar. You can introspect and modify live objects. (C)Python frees objects when their reference count hits zero; a cyclic garbage collector handles loops. Object death is deterministic—unless it’s part of a love triangle.

> [!info]- **Love triangles**
> Or **cyclic references**—closed loops where objects keep each other alive.
>
> ```python
> class Node:
>     def __init__(self, name): self.name, self.link = name, None
>
> a = Node("A")
> b = Node("B")
> c = Node("C")
>
> a.link = b
> b.link = c
> c.link = a  # cycle: A → B → C → A
> ```
>
> This **three-way reference** creates a cycle. None of the objects can be freed by reference counting alone—each one is kept alive by another.
>
> That’s when CPython’s **cyclic garbage collector** kicks in. It periodically scans for unreachable cycles and clears them out.
>
> So:
> - **Simple objects die predictably.**
> - **Objects tangled in a love triangle don’t die until the GC notices.**

> [!info]- **Everything is an object**
> ```python
> type(3)        # <class 'int'>
> type(len)      # <class 'builtin_function_or_method'>
> isinstance(None, object)  # True
> ```
> You can inspect or mutate attributes at runtime:
> ```python
> def f(): pass
> f.custom = 42
> print(f.custom)
> ```

> [!info]- **CPython**
> **CPython** is the **reference implementation** of Python—the one you’re almost certainly using unless you went out of your way not to.
> 
> It’s written in **C**, hence the name. When you type `python` or `python3` on most systems, you're running CPython.
> 
> There are other implementations:
> - **PyPy** — written in Python, faster in some cases (JIT-compiled).
> - **Jython** — runs on the Java Virtual Machine.
> - **IronPython** — runs on .NET.
> - **MicroPython** — for embedded systems.
> 
> But **CPython** is the original, the official, and the one all others are compared to.
> 
> When docs say “Python does X,” they usually mean **CPython does X**—because that’s what 99% of people are using.

#### **20. All variables are references—_including integers_**  
In Python, **variables don’t hold values**. They hold **references to objects**.  
Even simple assignments like `x = 5` just bind the name `x` to an existing object in memory.

> [!info]- **What this actually means**
>
> #### Assignment binds names to objects
> ```python
> x = 5
> y = x
> ```
> Both `x` and `y` point to the same object. There's no value copy here.
>
> #### Immutable, but still referenced
> Even though `int`, `str`, and `tuple` can’t be mutated, they’re still **objects bound by reference**, not copied.
>
> #### `is` vs `==`
> ```python
> a = 1000
> b = 1000
> a == b  # True — same value
> a is b  # False — usually different objects
> ```
> `==` checks if values are equal.
> `is` checks if both names point to the **same object.**
>
> #### CPython integer interning
> ```python
> x = 5
> y = 5
> x is y  # True (usually)
> ```
> CPython caches small integers (-5 to 256) for performance. That's why identity sometimes matches—but don't rely on it.
>
> ---
>
> #### C# comparison: value types vs references
> ```csharp
> int a = 5;
> int b = a;
> Console.WriteLine(a == b);  // True
> Console.WriteLine(Object.ReferenceEquals(a, b));  // False
> ```
> - C# `int` is a **value type**: assigning `b = a` copies the value.
> - In Python, **everything is a reference**, even for immutable types.
> - But in both languages, `int` is **immutable.**
>
> ---
>
> #### Mental model:
> - Names are **labels** stuck to objects.
> - Assignment moves the **label**, not the object.
> - Immutable types (e.g. `int`, `str`, `tuple`) can’t be changed.
> - Mutable types (`list`, `dict`) can be changed in place.

#### **21. The difference between `is` and `==` matters**  
`==` checks if two values are equal.  
`is` checks if two names point to the **same object in memory**.

> [!info]- **Equality vs identity**
>
> #### Example:
> ```python
> a = [1]
> b = [1]
> print(a == b)  # True — same contents
> print(a is b)  # False — different objects
> ```
>
> #### `is` means “same identity”:
> ```python
> x = a
> print(x is a)  # True — same object
> ```
>
> #### Use `is` for:
> - `None`: `if x is None:`  
> - `True` / `False` (sometimes)
>
> #### Don’t use `is` for:
> - Strings  
> - Numbers  
> - Containers  
>
> These may be **interned** or optimized under the hood:
> ```python
> a = "hi"
> b = "hi"
> print(a is b)  # Might be True — but not guaranteed
> ```
> Works today, fails tomorrow. Use `==` unless you **care about identity**.

#### **22. All arguments are passed by object-reference**  
Functions get a new name pointing to the same object—an **alias**. Mutate it, and changes stick. Rebind it, and nobody else notices—only the function cares. Mutables (`list`, `dict`) can be changed in-place, immutables (`int`, `str`, `tuple`) can’t.

> [!info]- **Elaboration**
> 
> Python passes **object references—by value**.
>
> - You’re not passing the actual object  
> - You’re not passing the caller’s variable  
> - You’re passing a **copy of the reference** (a second pointer for the same object)
> - This creates an **alias** inside the function
>
> ---
>
> #### Mutations **do** persist:
> If the object is mutable and the function modifies it, that change is shared.
>
> #### Reassignments **don’t** persist:
> If the function rebinds its parameter to a new object, the caller’s variable isn’t affected.
>
> ---
>
> #### Example:
> ```python
> def mutate(x):
>     x.append(99)  # edits the object
>
> def reassign(x):
>     x = [4, 5, 6]  # points x at a new object (only inside the function)
>
> a = [1, 2, 3]
> mutate(a)
> print(a)       # [1, 2, 3, 99] ← mutation persisted
>
> reassign(a)
> print(a)       # [1, 2, 3, 99] ← reassignment did *not* persist
> 
> a = [7, 8, 9]  # caller rebinds
> print(a)       # [7, 8, 9] — clean slate, no ties to the old list
> ```
>
> ---
>
> #### Analogy:
> Think of variables as **name tags** on objects.
> Python hands the function a **duplicate name tag**—an **alias**.  
> If the object gets modified, everyone sees it.  
> But if the function slaps its tag on a new object, you never know.

#### **23. Assignment doesn’t copy**—even if it _feels_ like it  
In Python, `a = b` doesn’t make a new object. It just gives `a` another reference to the same one. You only get a copy when you ask for one—explicitly.

> [!info]- **Reference vs copy**
>
> #### Names share memory:
> ```python
> a = [1, 2, 3]
> b = a
> b.append(4)
> print(a)  # [1, 2, 3, 4]
> ```
> `a` and `b` point to the same list. One change affects both.
>
> #### Shallow copies:
> ```python
> a = [1, 2, 3]
> b = a[:]         # slice copy
> c = list(a)      # constructor copy
> d = a.copy()     # method copy
> ```
> All create **new lists** that are shallow copies.
>
> #### In-place mutation:
> ```python
> a = [1, 2, 3]
> b = a
> a[:] = [9, 9, 9]
> print(b)  # [9, 9, 9]
> ```
> Looks like reassignment—but it's **mutation**, not a new object.
>
> #### Deep copies (nested structures):
> ```python
> import copy
> nested = [[1], [2]]
> shallow = list(nested)
> deep = copy.deepcopy(nested)
>
> nested[0][0] = 99
> print(shallow)  # [[99], [2]]
> print(deep)     # [[1], [2]]
> ```
> Shallow copies copy outer containers, not inner contents.  
> Use `copy.deepcopy()` if you want full duplication.
>
> ---
> 
> If you don’t want shared memory, **make it explicit**

#### **24. It’s dynamically typed, but strongly typed**  
You don’t declare types, but try `"5" + 1` and it’ll slap you. The freedom is great, but bugs hide until runtime. Static analysis tools like **`mypy`** or editors like **Pyright** can help, but they’re optional bolt-ons.

> [!info]- **Types**
> Python is **dynamically typed** and **strongly typed**.
>
> - **Dynamically typed** → You don’t declare types, and types are checked **at runtime**, not before.
> - **Strongly typed** → The interpreter **enforces type rules**. It won’t automatically convert types for you.
>
> #### Example
> ```python
> "5" + 1  # TypeError: can’t concatenate str and int
> ```
> You can’t mix types unless you explicitly convert:
> ```python
> int("5") + 1     # 6
> "5" + str(1)     # "51"
> ```
>
> ---
>
> #### Compare to JavaScript (dynamic + weak)
> ```javascript
> "5" + 1      // "51" — automatic coercion
> "5" - 1      // 4
> [] + {}      // "[object Object]"
> ```
> JS tries to guess what you meant. Python doesn’t guess.  
> If your types don’t match, Python throws immediately.
>
> ---
>
> #### Compare to C# (static + strong)
> ```csharp
> string s = "5";
> int i = 1;
> var result = s + i;  // Compile-time error (unless explicitly converted)
> ```
> In C#, you declare types up front.  
> The compiler catches type mismatches before your code runs.
>
> ---
>
> #### Python doesn't care
> ```python
> def shout(x):
>     return x.upper()
>
> shout(123)  # crashes at runtime
> ```
> - No type hints = no static checking.
> - `mypy` and `Pyright` have nothing to analyze.
> - Python runs it—and blows up at `123.upper()`.
>
> ---
>
> #### But you can opt in
> ```python
> def shout(x: str) -> str:
>     return x.upper()
>
> shout(123)  # runs, then crashes — no type enforcement
> ```
> Run this through **mypy** or **Pyright**, and they’ll catch the mismatch:
> ```
> error: Argument 1 to "shout" has incompatible type "int"; expected "str"
> ```

> [!info]- **Type hints**
>
> #### Type Hints in Python = TypeScript for JS
> - **JavaScript** = dynamic typing, no checks until runtime.
> - **TypeScript** = JS + static types + compile-time checks.
>
> - **Python** = dynamic typing, but **can be made safer**.
> - **mypy / Pyright** = like TypeScript compilers for Python.
>
> #### Example:
> ```python
> def shout(x: str) -> str:
>     return x.upper()
>
> shout(123)  # Crashes at runtime
> ```
> mypy or Pyright would catch that before it runs:
> ```
> error: Argument 1 to "shout" has incompatible type "int"; expected "str"
> ```
>
> ---
>
> #### Real-world usage:
> - In small scripts, type hints are optional.
> - In **production-grade** or **team-based** codebases, they're increasingly expected.
> - Tools like **FastAPI**, **Pydantic**, **VS Code (Pylance)**, etc. all leverage them.
> - Many teams run **mypy** or **Pyright** as part of their CI/CD pipeline.

> [!info]- **Modern Typing (Python 3.10+)**
> Python’s type system is evolving—but still optional.
>
> #### Union types:
> ```python
> def parse(x: int | str) -> int:
>     return int(x)
> ```
> Cleaner alternative to `Union[int, str]`.
>
> #### Structural pattern matching:
> ```python
> match value:
>     case 0:
>         print("zero")
>     case [a, b]:
>         print(f"two elements: {a}, {b}")
>     case _:
>         print("something else")
> ```
> Works like a smarter `switch`. Matches **structure**, not just values.
>
> #### Typed containers and generics:
> ```python
> from typing import List
>
> def total(xs: List[int]) -> int:
>     return sum(xs)
> ```
> Hints what kind of data is expected—checked by static analyzers.
>
> #### Type hint gotcha: circular imports
> If you're using lots of type annotations (especially with custom classes), add this at the *very top* of your file:
> ```python
> from __future__ import annotations
> ```
> This tells Python:  
> “Don't try to resolve type hints right now—wait until later.”  
>
> Why? Because without it, Python might try to evaluate a type **before** the thing is even defined.  
>
> - Needed in Python 3.10.
> - Default behavior in 3.11+.
> - Harmless to use always—just slap it on and move on.

#### **25. Everything is duck-typed**  
If it quacks like a list, it _is_ a list. Interfaces don’t matter—behavior does. This makes polymorphism effortless and debugging hellish. Start writing `hasattr` checks if you're feeling paranoid.

> [!info]- **Ducks**
> Python doesn’t care what an object *is*—only what it *does*. That’s duck typing:
> 
> > “If it walks like a duck and quacks like a duck... it’s a duck.”
> 
> In practice: if your object supports the right methods or behaviors, Python assumes it’s legit.
> 
> #### Example: works like a list, doesn’t matter what it is
> ```python
> def total_length(x):
>     return len(x)
> 
> total_length("hello")     # 5 (string)
> total_length([1, 2, 3])   # 3 (list)
> total_length((1, 2))      # 2 (tuple)
> ```
> No interfaces, no inheritance, no declarations. Just behavior.
> 
> ---
> 
> #### Custom duck
> ```python
> class FakeList:
>     def __len__(self):
>         return 42
> 
> print(len(FakeList()))  # 42
> ```
> Python doesn’t care that it’s not a list—it saw `__len__` and said “good enough.”
> 
> ---
> 
> #### The downside: you're guessing
> ```python
> def shout(x):
>     return x.upper()
> 
> shout("hello")       # "HELLO"
> shout(123)           # AttributeError: 'int' object has no attribute 'upper'
> ```
> Python didn’t warn you ahead of time. It trusted you. Duck *trusted* you. **If it can’t quack, it dies at runtime.**
> 
> ---
> 
> #### How to cope:
> - Use `hasattr(x, "method_name")` if unsure.
> - Use `try/except` to fail gracefully.
> - Use static type checkers (e.g. `mypy`) in larger projects.
> 
> Duck typing gives you flexibility. It also gives you just enough rope to decorate, trip, and accidentally lasso your own foot—all in one line.

> [!info]- **Further reading: Python’s interfaces**
>
> Duck typing is flexible—but if you want guarantees, Python gives you two formal tools:
>
> #### `typing.Protocol` (structural typing)
> Think TypeScript or Go-style interfaces.  
> A class doesn’t need to inherit—just match the shape.
> ```python
> from typing import Protocol
>
> class Flyer(Protocol):
>     def fly(self) -> None: ...
>
> class Bird:
>     def fly(self): print("flap")
>
> def launch(f: Flyer): f.fly()
>
> launch(Bird())  # OK
> ```
> No inheritance. No ceremony. Just behavior.
>
> #### `abc.ABC` (nominal typing)
> Like a C# abstract class or interface—must explicitly inherit and implement.
> ```python
> from abc import ABC, abstractmethod
>
> class Animal(ABC):
>     @abstractmethod
>     def speak(self): ...
>
> class Dog(Animal):
>     def speak(self): print("woof")
> ```
> Raises `TypeError` if methods aren’t implemented.
>
> ---
>
> Use `Protocol` when you want flexibility.  
> Use `ABC` when you want enforcement.  
> Or just trust the duck. Until it bites you.

## Mastering Functions & Control Flow
Python's powerful function semantics and the nuances of controlling program execution.

#### **26. Functions are first-class**  
They can be passed, returned, nested, stored. You’ll bump into `map`, `filter`, `lambda`, decorators, and partials. All of that’s just playing with function-as-object. You can write your own decorator before you learn what a class is.

> [!info]- **Function = Object**
> ```python
> def add(x, y): return x + y
> f = add        # stored in a variable
> print(f(1, 2)) # called via alias
> ```
>
> Functions can be passed to other functions:
>
> ```python
> def apply(fn, x): return fn(x)
> apply(abs, -5)  # 5
> ```
>
> And returned:
>
> ```python
> def make_adder(n):
>     return lambda x: x + n
> add_3 = make_adder(3)
> add_3(10)  # 13
> ```
>
> Decorators, lambdas, partials, callbacks—all ride on this.

#### **27. Decorators: The @ Syntax Unpacked**  
Decorators wrap a function with another function. They’re syntactic sugar for functional composition. Think of them like C# attributes that *run code*, not just annotate it.

> [!info]- **What decorators are**
>
> A decorator is a function that takes another function and returns a new one:
>
> ```python
> def loud(fn):
>     def wrapper(*args, **kwargs):
>         print("Calling", fn.__name__)
>         return fn(*args, **kwargs)
>     return wrapper
>
> @loud
> def greet():
>     print("Hi")
>
> greet()
> # Output:
> # Calling greet
> # Hi
> ```
>
> `@loud` is shorthand for `greet = loud(greet)`.
>
> ---
>
> #### Common Uses
> - **Logging**
> - **Authentication checks**
> - **Memoization** (`@lru_cache`)
> - **Framework routes** (Flask, FastAPI)
> - **Testing mocks** (`@patch`)
>
> You’ll use them before you understand them.

#### **28. Function defs have quirks**  
Default arguments are evaluated at _definition_ time, not runtime. Mutable defaults (e.g. `list=[]`) will trap you. Use `None` and assign inside.

> [!info]- **Default values are sticky**
>
> #### Evaluated once, not per call:
> ```python
> def foo(x=[]):
>     x.append(1)
>     return x
>
> foo()  # [1]
> foo()  # [1, 1]
> foo()  # [1, 1, 1]
> ```
> Python only evaluates `[]` once—when the function is *defined*, not each time it’s called.
>
> ---
>
> #### Why?
> Defaults live in the function’s `__defaults__` tuple. They’re static—baked into the function object.
>
> Good for this:
> ```python
> def connect(port=5432): ...
> ```
> Bad for this:
> ```python
> def log(msg, log_list=[]): ...
> ```
>
> ---
>
> #### The fix: use `None` as a sentinel
> ```python
> def foo(x=None):
>     if x is None:
>         x = []
>     x.append(1)
>     return x
> ```
> Now a new list is created each time.
>
> ---
>
> #### Applies to all mutable types:
> - `list`, `dict`, `set`.  
> - Any object you can mutate in-place.
>
> ---
>
> #### Rare case: shared default on purpose
> ```python
> def counter(start=0, _memo={}):
>     _memo[start] = _memo.get(start, 0) + 1
>     return _memo[start]
> ```
> This uses a shared default as internal storage. Rarely justified. Easy to misuse.

#### **29. No method overloading**  
Python doesn't support traditional method overloading where you can define multiple methods with the same name but different parameter types. The last definition of a function or method simply overwrites any previous ones.

> [!info]- **What happens if you try?**
>
> ```python
> def add(x, y):
>     return x + y
>
> def add(x, y, z):
>     return x + y + z
>
> add(2, 3)  # TypeError: missing 1 required positional argument
> ```
>
> The second `add()` replaces the first one—no warning, no error. Only the last one survives.
>
> ---
>
> The Pythonic solution is to use **default parameters** or **type checks** inside the function body:
>
> ```python
> def add(x, y, z=None):
>     if z is None:
>         return x + y
>     else:
>         return x + y + z
>
> add(2, 3)      # 5
> add(2, 3, 4)   # 9
> ```
>
> Or use type inspection:
>
> ```python
> def shout(x):
>     if isinstance(x, list):
>         return [str(i).upper() for i in x]
>     return str(x).upper()
> ```
>
> Or just accept `*args` and sort it out yourself.
>
> ---
>
> **C# comparison**
>
> In C#, you’d write:
>
> ```csharp
> int Add(int x, int y) => x + y;
> int Add(int x, int y, int z) => x + y + z;
> ```
>
> The compiler picks the matching signature.  
> In Python, the last one wins.
> (For advanced cases: look into `functools.singledispatch` for runtime dispatch, or `typing.overload` for static hints).

> [!info]- **Further reading**
>
> For advanced typing, look into `TypeVar`, `Generic`, and `Protocol` in the `typing` module. These allow you to express polymorphism and interface-like constraints—very useful in larger codebases.

#### **30. `*args` and `**kwargs` are Python’s escape hatches**  
Python lets you define functions that accept any number of positional and keyword arguments using `*args` and `**kwargs`. These aren’t just sugar—they're foundational for decorators, adapters, and APIs that need flexibility.

> [!info]- **What they actually mean**
>
> - `*args` collects **extra positional arguments** into a tuple.
> - `**kwargs` collects **extra keyword arguments** into a dict.
>
> Order matters:
>
> ```python
> def func(pos1, pos2, *args, kw1=None, **kwargs): ...
> ```
>
> This lets you mix fixed arguments, flexible positionals, named keyword-only args, and open-ended keyword args—all in one function.
>
> ---
>
> #### Example:
>
> ```python
> def kitchen_sink(a, b, *args, **kwargs):
>     print(f"a = {a}")
>     print(f"b = {b}")
>     print(f"Extra positionals: {args}")
>     print(f"Extra keywords: {kwargs}")
>
> kitchen_sink(1, 2, 3, 4, name="Alice", age=30)
> ```
>
> Output:
> ```
> a = 1
> b = 2
> Extra positionals: (3, 4)
> Extra keywords: {'name': 'Alice', 'age': 30}
> ```
>
> ---
>
> #### Usage patterns:
>
> - Wrapping any unknown function signature.
> - Writing decorators.
> - Adapting legacy APIs.
> - Logging/monitoring calls.
>
> ---
>
> #### Argument unpacking (the flip side):
>
> You can **pass** a tuple/dict as `*args` and `**kwargs` too:
>
> ```python
> def greet(name, age): ...
>
> args = ("Alice", 30)
> kwargs = {"name": "Bob", "age": 40}
>
> greet(*args)
> greet(**kwargs)
> ```
>
> ---
>
> #### C# comparison:
>
> - C# supports **named arguments**, but no true `**kwargs`.
> - You’d use `params` for variadics, and overloads or dictionaries for keyword flexibility.
>
> Python bakes it in.

#### **31. Keyword-only arguments clarify intent**  
Python lets you force some arguments to be passed **only by name**, using a single `*` in the function signature. This makes function calls more readable and avoids position-based confusion.

> [!info]- **How it works**
>
> ```python
> def move(x, y, *, speed=1):
>     print(f"Moving to ({x},{y}) at speed {speed}")
>
> move(5, 10, speed=2)   # ✅ okay
> move(5, 10, 2)         # ❌ TypeError
> ```
>
> - Any parameter **after** `*` must be specified using a keyword.
> - You can combine this with regular, defaulted, or variadic arguments.
>
> ---
>
> #### When it matters:
>
> - When your function has lots of optional arguments.
> - When position is ambiguous.
> - When you're writing libraries or APIs.
>
> ---
>
> #### Also works with `*args`
>
> ```python
> def log(msg, *args, severity="info"):
>     print(f"[{severity.upper()}] {msg % args}")
> ```
>
> Forces `severity` to be named—can’t sneak in as a positional.
>
> ---
>
> Enforcing keyword-only arguments improves clarity and prevents bugs. Use it like a contract.

#### **32. Multiple return values via tuples**  
Python functions can return multiple values by returning a tuple. This is so common that unpacking returned tuples into separate variables is considered idiomatic. It keeps code clean and expressive.

> [!info]- **Returning multiple values**
>
> ```python
> def get_user_info(user_id):
>     # Pretend this queries a database
>     name = "Preben"
>     age = 42
>     return name, age  # Implicitly returns a tuple: ('Preben', 42)
>
> user_name, user_age = get_user_info(1)
> print(f"{user_name} is {user_age} years old.")
> # Preben is 42 years old.
> ```
>
> ---
>
> #### Unpacking works anywhere:
>
> ```python
> x, y = (1, 2)
> a, b, c = [10, 20, 30]
> first, *rest = [1, 2, 3, 4]
> ```
>
> Python's assignment model treats unpacking as natural—because **tuples are just sequences**.
>
> ---
>
> #### You can return anything:
>
> ```python
> def split_name(full):
>     return full.split(" ", 1)
>
> first, last = split_name("Ada Lovelace")
> ```
>
> No named tuple needed. Just return values.
>
> ---
>
> #### C# comparison:
>
> C# 7+ allows tuple returns:
>
> ```csharp
> (string name, int age) GetUser() => ("Preben", 42);
> var (name, age) = GetUser();
> ```
>
> But in Python, this behavior is **the norm**, not an advanced feature. Functions returning multiple values is standard practice—not a workaround.

#### **33. Truthiness is fuzzy**  
Empty containers are falsy. Be explicit when you care (`is None`), not just `if x`.

> [!info]- **Elaboration**
> In Python, a lot of things count as “falsey”:
> 
> - `0`, `0.0`, `''`, `[]`, `{}`, `set()`, `None`, `False`, `range(0)`
> - Even custom objects, if they define `__bool__` or `__len__` returning `False` or `0`
> 
> So when you write:
> ```python
> if x:
>     do_something()
> ```
> You’re not checking “is `x` True?”, you’re checking “is `x` truthy?”
> 
> ---
> 
> #### Sometimes that’s fine:
> ```python
> if my_list:
>     # List has at least one item
> ```
> 
> ---
> 
> #### But sometimes it’s dangerously vague:
> ```python
> def get_user_age():
>     return 0  # newborn
> 
> if not get_user_age():
>     print("No age provided.")  # Wrong
> ```
> `0` is a valid age—but it’s also falsey.
> 
> ---
> 
> #### Bottom line:
> Be explicit when it matters.  
> Use `if x is None`, `if len(x) == 0`, or `if x == 0`—not just `if x`—when precision matters.

#### **34. The walrus operator assigns in expressions**  
The `:=` operator (Python 3.8+) lets you assign values **as part of an expression**. Great for avoiding repetition, and surprisingly readable when used right.

> [!info]- **`:=`**
>
> ```python
> if (n := len(data)) > 10:
>     print(f"{n} items")
> ```
>
> - `n` gets assigned **inside** the `if` condition.
> - The expression both **assigns** and **evaluates**.
>
> ---
>
> #### Without walrus:
>
> ```python
> n = len(data)
> if n > 10:
>     ...
> ```
>
> Walrus avoids the extra line. Use when it **improves clarity**, not just to be clever.
>
> ---
>
> #### Other examples:
>
> ```python
> while (line := file.readline()) != "":
>     print(line) # Stops on empty string (EOF)
> ```
>
> ```python
> if (match := re.search(pattern, text)):
>     print(match.group(1))
> ```
>
> ---
>
> Not required—but when you want it, nothing else quite does the trick.
>
> _(Also responsible for one of Python’s pettiest civil wars. Look it up.)_

#### **35. `and` / `or` return values, not just `True` or `False`**  
Python’s boolean operators don’t force a True/False. They return the **actual value** of the last thing they checked. This lets you chain logic _and_ extract values in one go—but it also trips up newcomers expecting a clean boolean.

> [!info]- **Explanation with examples**
>
> Python uses **short-circuit evaluation**:
>
> - `a or b` returns `a` if it's truthy; otherwise `b`
> - `a and b` returns `a` if it's falsy; otherwise `b`
>
> #### Examples:
> ```python
> "" or "fallback"         # "fallback"
> "hi" or "fallback"       # "hi"
>
> 0 and 5                  # 0
> 3 and 5                  # 5
> None or [] or {} or 42   # 42
> ```
>
> ---
>
> #### Why it matters:
> These can be used for clever fallbacks:
> ```python
> result = user_input or default_value
> ```
> But if you're expecting a strict `True`/`False`, wrap it:
> ```python
> bool(x and y)
> ```

#### **36. `match` is not a switch**  
Python 3.10+ introduces `match`—but it’s not C-style. It’s **pattern matching**, more like Rust or functional languages. It matches structure, not just values.

> [!info]- **Basic usage**
>
> ```python
> command = ["move", 10, 20]
>
> match command:
>     case ["move", x, y]:
>         print(f"Moving to ({x}, {y})")
>     case ["stop"]:
>         print("Stopping")
>     case _:
>         print("Unknown command")
> ```
>
> - Matches structure and binds variables.
> - Supports destructuring, guards, type checks.
>
> ---
>
> #### With guards:
>
> ```python
> match x:
>     case int() as i if i > 0:
>         print("positive int")
> ```
>
> ---
>
> #### With objects:
>
> ```python
> class Point: ...
>
> match obj:
>     case Point(x=0, y=y):
>         print("On Y axis")
> ```
>
> ---
>> [!warning] **`match` stops at the first match**
>> Python checks each `case` **top to bottom**, and **runs only the first one that matches**. There’s no fall-through like in C-style `switch` statements.
>>
>> ```python
>> match x:
>>     case int():
>>         ...      # This runs for *any* int
>>     case int() if x > 0:
>>         ...      # Never reached
>> ```
>>
>> So:
>> - Put **more specific cases first**.
>> - Use `if` guards when needed.
>> - Don't assume later cases will run.
> 
> `match` makes branching logic readable and expressive—when structure matters.

## Data Structures & Iteration
Built-in collections and powerful tools for iterating over and manipulating sequences of data.

#### **37. Core collection types: list, tuple, dict, set**  
Python has four built-in data structures you’ll use constantly. Each has specific behavior around **mutability**, **ordering**, and **uniqueness**. Know these cold.

> [!info]- **Summary table**
>
> | Type  | Ordered? | Mutable? | Allows duplicates? | C# Analog         |
> |-------|----------|----------|---------------------|-------------------|
> | list  | Yes      | Yes      | Yes                | `List<T>`         |
> | tuple | Yes      | No       | Yes                | `Tuple` / readonly array |
> | dict  | Yes (3.7+)| Yes     | Keys unique        | `Dictionary<TKey, TValue>` |
> | set   | No       | Yes      | No (all unique)    | `HashSet<T>`      |
> __
> **Note**:
> - `dict` preserves insertion order as part of the language spec (since 3.7).
> - `set` **also preserves insertion order in CPython 3.7+**, but this is an **implementation detail**, **not guaranteed by the spec**. It may look ordered—**don’t write code that depends on it**.
> ---
>
> #### `list`: the mutable workhorse
>
> ```python
> my_list = [1, "a", 3.0]
> my_list.append(4)
> my_list[1] = "b"
> ```
>
> - Ordered
> - Mutable
> - Allows duplicates
>
> Use when you need a sequence that can grow, shrink, or mutate.
>
> ---
>
> #### `tuple`: like a list, but frozen
>
> ```python
> my_tuple = (1, "a", 3.0)
> # my_tuple[0] = 99  # TypeError
> ```
>
> - Ordered
> - Immutable
> - Allows duplicates
>
> Use for fixed collections, coordinates, or multiple returns.
>
> ---
>
> #### `dict`: key-value mapping
>
> ```python
> my_dict = {"name": "Preben", "age": 42}
> my_dict["age"] += 1
> ```
>
> - Ordered (Python 3.7+ — now part of the language spec).
> - `popitem()` pops the **last inserted** item (LIFO) since Python 3.8.
> - Mutable.
> - Keys must be hashable (str, int, tuple, etc.).
>
> Use for fast lookups and key-value data.
>
> ---
>
> #### `set`: unique unordered collection
>
> ```python
> my_set = {1, 2, 3}
> my_set.add(2)     # No effect—already present
> my_set.add(4)     # Adds 4
> ```
>
> - Unordered (even if it looks ordered in CPython—don’t rely on it).
> - Mutable.
> - No duplicates.
>
> Use for deduplication, membership tests, and set operations.
>
> ---
>
> #### Set operations
>
> ```python
> a = {1, 2, 3}
> b = {3, 4, 5}
>
> a | b   # union: {1, 2, 3, 4, 5}
> a & b   # intersection: {3}
> a - b   # difference: {1, 2}
> ```
>
> Deduplicate a list:
> ```python
> unique = list(set([1, 2, 2, 3]))
> ```

#### **38. Everything’s iterable, but not everything's an iterator**  
Python loops don’t check types—they check for `__iter__` and `__next__`.  
Strings, lists, dicts: iterable.  
Files, generators: iterators.  
Know the difference. `iter()` gets an iterator from an iterable. `next()` walks it. Once an iterator is exhausted, it’s dead. Generators (`yield`) are one-shot conveyor belts.

> [!info]- **Iterable vs Iterator**
>
> #### Iterable:
> - Any object that can be looped over.
> - Must implement `__iter__()`, which returns an **iterator.**
> - Examples: `list`, `str`, `dict`, `set`, `range`.
>
> #### Iterator:
> - Implements both `__iter__()` and `__next__()`.
> - Produces values one at a time with `next()`.
> - Remembers its position.
> - Dies when exhausted (raises `StopIteration`).
>
> ---
>
> #### Example:
> ```python
> xs = [1, 2, 3]
> it = iter(xs)
> next(it)  # 1
> next(it)  # 2
> next(it)  # 3
> next(it)  # StopIteration
> ```
>
> ---
>
> #### Generator = custom iterator:
> ```python
> def countdown(n):
>     while n > 0:
>         yield n
>         n -= 1
>
> c = countdown(3)
> list(c)  # [3, 2, 1]
> list(c)  # [] — already exhausted
> ```
>
> Generators are **lazy**: they pause between values.  
> And **one-shot**: once they're done, they're done.
>
> ---
>
> **C# comparison**  
> Generators (`yield`) are Python’s equivalent of `IEnumerable<T>` with `yield return`:  
> lazy sequences that generate values on-demand, without materializing a list.
>
> **Rule of thumb:**  
> If you can loop over it, it's iterable.  
> If it remembers where it left off, it's an iterator.

#### **39. Generators aren’t just output—they can take input**  
`yield` pauses execution, `.send()` resumes it *with a value*.  
Before `async`, this was coroutine country.

> [!info]- **Injecting values into a generator**
> ```python
> def echo():
>     received = yield "start"
>     while True:
>         received = yield f"got: {received}"
>
> g = echo()
> print(next(g))          # start
> print(g.send("hi"))     # got: hi
> print(g.send("again"))  # got: again
> ```

#### **40. `itertools` = black-belt list magic**  
It’s the kitchen drawer of composable iteration tools.  
`chain`, `product`, `combinations`, `cycle`, `islice`, `groupby`, etc.  
Don’t reinvent what’s already in this module. It’s a cheat code.

> [!info]- **Examples of `itertools` in action**
> ```python
> from itertools import chain, product, islice
> 
> a = [1, 2]
> b = [3, 4]
> list(chain(a, b))              # [1, 2, 3, 4]
> 
> list(product("AB", repeat=2)) # [('A','A'), ('A','B'), ('B','A'), ('B','B')]
> 
> it = iter(range(10))
> list(islice(it, 3, 7))         # [3, 4, 5, 6]
> ```

#### **41. Scoping of list comps is not what you'd expect**  
In Python 3, list comprehensions have their **own scope**.  
Python 2 leaked the loop variable into the outer scope.

`x = 5`  
`print([x for x in range(3)])  # [0, 1, 2]`  
`print(x)                      # still 5`

> [!info]- **Python 2 was different (and dangerous)**
> In Python 2, the loop variable from a list comp **leaked** into the outer scope.
> ```python
> x = 5
> print([x for x in range(3)])  # [0, 1, 2]
> print(x)                      # 2 — x got overwritten
> ```
> This behavior was fixed in Python 3.  If you're stuck maintaining legacy code, trust nothing, assume sabotage.

#### **42. Starred unpacking lets you grab “the rest” of a sequence**  
You can use `*` to catch multiple values during assignment or pass elements into functions.  
It’s powerful—but brittle if you don’t know the shape of your data.

> [!info]- **Starred unpacking**
>
> #### In assignments:
> ```python
> a, *b, c = [1, 2, 3, 4, 5]
> print(a)  # 1
> print(b)  # [2, 3, 4]
> print(c)  # 5
> ```
> `*b` grabs everything between the first and last elements.
>
> #### You can use it in loop targets too:
> ```python
> for a, *b in [(1, 2, 3), (4, 5, 6)]:
>     print(a, b)
> # 1 [2, 3]
> # 4 [5, 6]
> ```
>
> #### In literals:
> ```python
> a = [1, 2]
> b = [3, 4]
> merged = [*a, *b]
> print(merged)  # [1, 2, 3, 4]
> ```
>
> #### In function calls:
> ```python
> def add(x, y): return x + y
> args = (2, 3)
> print(add(*args))  # 5
> ```
>
> #### With dictionaries:
> `**` works for unpacking key-value pairs, **but only in function calls or dict literals**:
> ```python
> def show(**kwargs): print(kwargs)
> show(**{"x": 1, "y": 2})  # {'x': 1, 'y': 2}
>
> d = {**{"a": 1}, **{"b": 2}}  # {'a': 1, 'b': 2}
> ```
>
> ---
>
> #### But you can’t do this:
> ```python
> **x = {"a": 1}  # SyntaxError
> ```
> `**` is not allowed in variable unpacking—only `*` is.
>
> ---
>Python must be able to predict how many values land where—too vague = crash.

#### **43. Integer division changed—Python 2 fossils still roam**

In **Python 3**:
- `5 / 2` → `2.5` (float)
- `5 // 2` → `2` (floor division)

In **Python 2**:
- `5 / 2` → `2` (truncating integer division)
- So `from __future__ import division` was common in transitional code.
Still shows up in dusty codebases and outdated tutorials. Don’t be fooled.

## The Object Model: Classes & Magic
Python's object-oriented features, the "dunder" methods that hook directly into the language's syntax.

#### **44. Explicit `self` in Methods**  
In C#, the `this` keyword is an implicit reference to the current instance. In Python, you must explicitly declare the instance reference as the first argument of any instance method, conventionally named `self`. It's not a keyword, just a strong convention.

> [!info]- **What is `self` and how does it compare to `this`?**
>
> - In C#/Java: `this` is **implicit**.
> - In Python: `self` is **explicit**.
>
> The method call `my_dog.bark()` is shorthand for:
>
> ```python
> Dog.bark(my_dog)
> ```
>
> Python **passes the instance as the first argument** behind the scenes. You have to name it explicitly—usually `self`. It’s not a keyword, just convention. You could call it `this`, `me`, `hamburger`, etc.—but don’t.
>
> ```python
> class Dog:
>     def bark(this):
>         print("Woof!", this)
> ```
>
> That works. It also marks you as unhinged.
>
> ---
>
> #### Comparison to C#
>
> ```csharp
> class Dog {
>     public void Bark() {
>         Console.WriteLine("Woof!");
>     }
> }
>
> var dog = new Dog();
> dog.Bark();  // 'this' is implicit
> ```
>
> ```python
> class Dog:
>     def bark(self):
>         print("Woof!")
>
> dog = Dog()
> dog.bark()  # 'self' is passed automatically
> ```
>
> ---
>
> #### Common mistake: forgetting `self`
>
> ```python
> class Dog:
>     def bark():  # forgot self
>         print("woof")
>
> Dog().bark()  # TypeError
> ```
>
> Python expected one argument (`self`) and didn’t get it.
>
> ---
>
> #### Why this design?
>
> - Python keeps method calls honest.
> - `self` makes the object reference visible.
> - Methods are just functions that happen to live in a class.
> - You’re not in OOP jail—Python doesn’t force object-orientation.

#### **45. Dunder methods = secret doors**  
`__init__`, `__str__`, `__repr__`, `__getitem__`, `__call__`, `__enter__/__exit__`, etc. These aren't just style quirks—they _hook into the language itself_. If you implement `__iter__`, Python treats your object like a loop. It’s black magic, but it’s documented black magic.

> [!info]- **Explanation**
> Dunder methods (double underscore methods) are Python’s built-in **hooks**. Implement one, and Python automatically wires your object into its syntax.
> 
> #### Example: build a custom object with dunder methods
> ```python
> class Thing:
>     def __init__(self, name):
>         self.name = name                
>         # called when you do Thing("something")
>     
>     def __str__(self):
>         return f"Thing: {self.name}"    
>         # called by print(t)
> 
>     def __len__(self):
>         return 42                       
>         # called by len(t)
> 
>     def __getitem__(self, key):
>         return f"Accessed key {key}"    
>         # called by t[key]
> ```
> 
> #### Now try it:
> ```python
> t = Thing("gadget")
> print(t)           # Thing: gadget         (__str__)
> len(t)             # 42                    (__len__)
> t["whatever"]      # 'Accessed key whatever' (__getitem__)
> ```
> These methods trigger automatically when you use core Python syntax. You don’t call them directly—they’re called *for* you.
> 
> ---
> 
> #### Common dunder hooks:
> - `__init__` — object creation  
> - `__str__` — friendly string (e.g. `print(x)`)  
> - `__repr__` — unambiguous debug string (`x` in REPL)  
> - `__len__`, `__bool__` — length, truthiness  
> - `__getitem__`, `__setitem__` — indexing and assignment  
> - `__call__` — make object behave like a function  
> - `__enter__`, `__exit__` — enable use in `with` blocks  
> - `__iter__`, `__next__` — enable iteration (e.g. `for x in obj`)
> 
> ---
> 
> #### C# comparison: overriding `ToString()`
> In C#:
> ```csharp
> public class Thing {
>     public string Name;
>     public Thing(string name) {
>         Name = name;
>     }
> 
>     public override string ToString() {
>         return $"Thing: {Name}";
>     }
> }
> 
> var t = new Thing("gadget");
> Console.WriteLine(t);  // Thing: gadget
> ```
> 
> In Python:
> ```python
> class Thing:
>     def __init__(self, name):
>         self.name = name
>     def __str__(self):
>         return f"Thing: {self.name}"
> 
> print(Thing("gadget"))  # Thing: gadget
> ```
> Both override a built-in method (`ToString` / `__str__`) that gets called implicitly by the language.
> 
> ---
> 
> #### Why it matters:
> These methods let you **define how your object behaves in core Python syntax**—printing, indexing, loops, logic, cleanup. They aren’t just optional decorations. They’re how Python *talks* to your object.

> [!info]- **`__repr__` vs `__str__`: Know the Difference**
>
> Python uses two special methods to represent objects as strings:
>
> - `__repr__`: Unambiguous and dev-facing. Used in the REPL, `repr()`, and fallback for `str()`.
> - `__str__`: Pretty-printing for users. Called by `print()` and `str()`.
>
> ```python
> class User:
>     def __init__(self, name): self.name = name
>
>     def __repr__(self): return f"User('{self.name}')"
>     def __str__(self): return self.name
>
> u = User("Preben")
> print(str(u))   # Preben
> print(repr(u))  # User('Preben')
> ```
>
> ---
>
> - If `__str__` is missing, Python uses `__repr__`.
> - If *both* are missing, you get a generic object string.
> - Good practice: always define `__repr__`, optionally override `__str__`.
>
> C# equivalent? `ToString()` is closest, but Python separates debug vs. display.

#### **46. `__slots__` = object memory control**  
By default, Python objects store attributes in a per-instance `__dict__`, which is flexible but memory-heavy.  
Defining `__slots__` in a class removes the `__dict__` and restricts which attributes can exist—saving memory, but limiting dynamism.

> [!info]- **How `__slots__` works**
>
> #### Normal class (with `__dict__`)
> ```python
> class Point:
>     def __init__(self, x, y):
>         self.x = x
>         self.y = y
> 
> p = Point(1, 2)
> print(p.__dict__)  # {'x': 1, 'y': 2}
> ```
> Each instance gets a dynamic dictionary for attributes. Flexible, but costly.
>
> ---
>
> #### With `__slots__`
> ```python
> class Point:
>     __slots__ = ("x", "y")
>
>     def __init__(self, x, y):
>         self.x = x
>         self.y = y
> 
> p = Point(1, 2)
> print(p.__slots__)  # ('x', 'y')
> ```
> - No `__dict__` per instance.
> - Fixed set of allowed attributes.
> - Saves memory (especially in large batches of objects), but no longer significantly speeds up attribute access (as of Python 3.11+).
>
> ---
>
> #### Limitations:
> - Can’t add new attributes outside of `__slots__`.
> - Can’t use features that rely on `__dict__` (e.g. `vars()`).
> - Doesn’t play nicely with multiple inheritance unless carefully managed.
>
> ---
>
> Use `__slots__` when:
> - You’re creating **lots** of instances.
> - Attribute names are known and fixed.
> - You care about **memory** or **attribute access speed**.

#### **47. `dataclasses` are cheat-mode for boilerplate**  
Added in Python 3.7, `@dataclass` auto-generates constructor, `__repr__`, and comparison methods. It’s a lightweight way to make immutable (or mutable) value objects.

> [!info]- **Basic example**
>
> ```python
> from dataclasses import dataclass
>
> @dataclass
> class User:
>     name: str
>     age: int
>
> u = User("Preben", 42)
> print(u.name)     # Preben
> print(u)          # User(name='Preben', age=42)
> ```
>
> ---
>
> #### Features:
>
> - Auto-generated `__init__`, `__repr__`, `__eq__`, etc.  
> - Optional immutability (`frozen=True`).
> - Default values and `field()` customization.
> - Works with type hints, IDEs, and static checkers.
>
> ---
>
> #### C# comparison:
>
> ```csharp
> public record User(string Name, int Age);
> ```
>
> - Python: `@dataclass`.
> - C#: `record`.
> - Same concept: value-type semantics without boilerplate.
>
> ---
>
> Use `@dataclass` for clean, honest data containers without manual constructor hell.

#### **48. Python has real enums—but you’ll almost forget they exist**  
The `enum` module provides class-based enumerations with named constants. They're safer than strings or integers and integrate cleanly into modern code—but most scripts still just wing it.

> [!info]- **Basic usage**
>
> ```python
> from enum import Enum
>
> class Color(Enum):
>     RED = 1
>     GREEN = 2
>     BLUE = 3
>
> print(Color.RED)         # Color.RED
> print(Color.RED.name)    # 'RED'
> print(Color.RED.value)   # 1
> ```
>
> ---
>
> #### Enums are type-safe:
>
> ```python
> def paint(c: Color):
>     if c == Color.BLUE:
>         print("Painting blue")
>
> paint(Color.GREEN)
> ```
>
> Comparisons are strict—`Color.RED != 1`.
>
> ---
>
> #### Iteration and lookup:
>
> ```python
> for color in Color:
>     print(color.name, color.value)
>
> Color["RED"]      # Color.RED
> Color(1)          # Color.RED
> ```
>
> ---
>
> #### Advanced: auto values and mixins
>
> ```python
> from enum import auto
>
> class State(Enum):
>     START = auto()
>     RUNNING = auto()
>     STOPPED = auto()
> ```
>
> Add mixins for ordering, string conversion, etc.
>
> ---
>
> #### C# comparison:
>
> C# has built-in enum support with better syntax:
>
> ```csharp
> enum Color { Red = 1, Green = 2, Blue = 3 }
> ```
>
> Python enums are more verbose, but far more flexible at runtime.

#### **49. Python objects can be context managers**  
Python’s `with` statement is the direct equivalent of C#’s `using` block:  
deterministic setup and teardown of resources like files, sockets, and locks.  
It’s based on two special methods: `__enter__` and `__exit__`.

Implement them, and your object works in a `with` block.  
This enables automatic cleanup—no `finally`, no leaks.

> [!info]- **Context managers in action**
>
> Any object with `__enter__` and `__exit__` can be used in a `with` block:
>
> ```python
> class Door:
>     def __enter__(self):
>         print("Door opens")
>         return self
>     def __exit__(self, exc_type, exc_val, exc_tb):
>         print("Door closes")
>
> with Door():
>     print("Inside")
> ```
> Output:
> ```
> Door opens
> Inside
> Door closes
> ```
>
> ---
>
> #### Real-world uses
> - **Files**: `with open(...)` auto-closes files.
> - **DB**: wrap transactions and roll back on failure.
> - **Threads**: `with lock:` to acquire/release safely.
> - **Testing**: `unittest.mock.patch` is a context manager.
> - **Temp files, sockets, timers, logging...**
>
> ---
>
> #### Even cleaner: `contextlib`
>
> Use `contextlib.contextmanager` to write one as a generator:
>
> ```python
> from contextlib import contextmanager
>
> @contextmanager
> def whisper():
>     print("...hush...")
>     yield
>     print("...done...")
>
> with whisper():
>     print("doing crimes")
> ```

#### **50. Descriptors power properties**  
`@property` isn’t magic—it’s just a wrapper around a deeper protocol: `__get__`, `__set__`, and `__delete__`. That’s how things like `staticmethod`, `classmethod`, and Django model fields work. You can write your own, and in some projects, you’ll need to.

> [!info]- **How descriptors work**
>
> A **descriptor** is any object that defines at least one of:
>
> - `__get__(self, instance, owner)`
> - `__set__(self, instance, value)`
> - `__delete__(self, instance)`
>
> Python calls these automatically when you access the attribute it’s bound to:
>
> ```python
> class Reveal:
>     def __get__(self, instance, owner):
>         print("Access granted")
>         return 42
>
> class MyClass:
>     x = Reveal()
>
> obj = MyClass()
> obj.x  # triggers Reveal.__get__
> ```
>
> ---
>
> #### `@property` is just a descriptor
>
> ```python
> class Celsius:
>     def __init__(self, temp): self._temp = temp
>
>     @property
>     def temp(self): return self._temp
>
>     @temp.setter
>     def temp(self, value): self._temp = value
> ```
>
> `@property` creates a descriptor under the hood. That’s why it can intercept reads and writes like a method—but behave like an attribute.
>
> ---
>
> #### C# comparison:
>
> In C#, properties are built into the language:
>
> ```csharp
> public class Celsius {
>     private int _temp;
>     public int Temp {
>         get { return _temp; }
>         set { _temp = value; }
>     }
> }
> ```
>
> Python doesn’t have native property syntax at the language level—`@property` just builds a descriptor object behind the scenes. But the outcome is the same: method logic with attribute syntax.
>
> ---
>
> #### Why care?
>
> - Used by `@staticmethod`, `@classmethod`, `@property`.
> - Core to **Django**, **SQLAlchemy**, and other frameworks.
> - Enables computed attributes, validation logic, mock behaviors.  
>
> You don’t need it often. But when you do, there’s no substitute.

#### **51. Metaclasses exist**  
They control _class creation_, not instance creation. You can hook into `__new__`, `__init__`, and mutate the class itself. They are how frameworks like Django do magic. Avoid until absolutely needed.

> [!info]- **What is a metaclass—and why should you fear it?**
>
> A **metaclass** is to a class what a class is to an instance.  
> When Python builds a class, it calls the metaclass to do it.
>
> ```python
> class Meta(type):
>     def __new__(cls, name, bases, dct):
>         print(f"Building class {name}")
>         return super().__new__(cls, name, bases, dct)
>
> class Thing(metaclass=Meta):
>     pass
> ```
> Output:
> ```
> Building class Thing
> ```
>
> ---
>
> #### Why use them?
> - Enforce class structure or constraints.
> - Automatically register subclasses.
> - Inject methods or attributes.
> - Modify decorators, base classes, docstrings, etc.
>
> Django’s ORM uses metaclasses to turn your class definitions into SQL field mappings.  
> Other frameworks use them for plugins, schemas, or declarative syntax.
>
> ---
>
> #### Rule of thumb
> If you *think* you need one, you probably don’t. If you _really_ need one, you’ll know.

> [!info]- **Lesser reading**
>
> There’s also `__metaclass__`, a legacy holdover from Python 2. Avoid it. You’ve already seen the modern approach.

#### **52. Class vs instance vs static methods**  
The three flavors of methods are weird until you see how `@classmethod`, `@staticmethod`, and `self` change behavior.

> [!info]- **Explanation**
> Python has three kinds of methods, and the difference is in **what gets passed as the first argument**:
> 
> ---
> 
> #### Instance method (default)
> ```python
> class Dog:
>     def bark(self):
>         print("bark", self)
> 
> dog = Dog()
> dog.bark()  # calls bark(dog)
> ```
> - `self` = the **instance**.
> - This is what you get by default—most methods are instance methods.
> - Used when the method works on instance-specific data.
> 
> ---
> 
> #### Class method
> ```python
> class Dog:
>     count = 0
> 
>     @classmethod
>     def make(cls):
>         cls.count += 1
>         return cls()
> 
> dog = Dog.make()
> ```
> - `cls` = the **class**, not an instance.
> - `cls()` = **call the class to make a new instance**.
> - `make()` here is a **factory method**: it tracks how many dogs were made, then returns a fresh one.
> - Unlike `Dog()`, `cls()` respects inheritance—if `Poodle(Dog)` overrides `make()`, `cls()` will be `Poodle`, not `Dog`.
> 
> ---
> 
> #### Static method
> ```python
> class Dog:
>     @staticmethod
>     def wag():
>         print("wagging")
> 
> Dog.wag()
> ```
> - No `self`, no `cls`—just a normal function living inside the class.
> - Useful for utility functions that conceptually belong to the class but don’t touch it.
> 
> ---
> 
> #### Summary
> | Type           | First arg | Access to instance? | Access to class? |
> |----------------|-----------|---------------------|------------------|
> | instance       | `self`    | ✅                  | ✅ via `type(self)` |
> | class          | `cls`     | ❌                  | ✅               |
> | static         | _none_    | ❌                  | ❌               |
> 
> ---
> 
> #### When you’ll misuse them:
> - You’ll forget `self` or `cls` and get a cryptic TypeError.
> - You’ll use `@staticmethod` when you really needed to mutate class or instance state.
> - You’ll write `Dog()` instead of `cls()` and break inheritance.
> - You’ll wonder why Python even has all three—until it clicks.

#### **53. Errors are objects**  
You can define your own exceptions. They can have arguments. You can raise them like `raise MyCustomError("You fool!")`. Stack traces are inspectable. You can catch multiple types in one line.

> [!info]- **Example**
>
> #### Custom exceptions are just classes:
> ```python
> class RageError(Exception):
>     def __init__(self, message, level):
>         super().__init__(message)
>         self.level = level
> ```
>
> #### You can raise them with arguments:
> ```python
> raise RageError("You absolute clown", level=9000)
> ```
>
> #### And catch them cleanly:
> ```python
> try:
>     do_something()
> except (ValueError, RageError) as e:
>     print(f"Problem: {e}")
>     if isinstance(e, RageError):
>         print(f"Rage level: {e.level}")
> ```
>
> ---
>
> #### Why this matters:
> You can use exceptions to **encode logic**, not just crash.  
> Want to interrupt control flow? Raise. Want context? Pass data.
> 
> If you’re still returning `"ERROR"` strings, you’re not really writing Python.

## The Broader Universe: Modules & Libraries
How code is organized and the powerful standard library and the third-party ecosystem.

#### **54. Namespaces are layered, not flat**  
LEGB: Local, Enclosing, Global, Built-in.  
When you reference a name, Python climbs this ladder. Closures live in the "Enclosing" scope, and `global`/`nonlocal` are how you punch through it—**if you dare**. It’s not _just_ variable shadowing; it defines what gets captured and **mutated**.

> [!info]- **LEGB Rule: Scope Resolution Order**
> - **Local**: Names defined inside the current function.
> - **Enclosing**: Names in enclosing function scopes (used by closures).
> - **Global**: Names at the top-level of the module.
> - **Built-in**: Names in the built-in namespace (`len`, `range`, etc.).
>
> ```python
> def outer():
>     x = "enclosing"
>     def inner():
>         print(x)  # finds x in the enclosing scope
>     inner()
> ```
>
> `nonlocal` lets inner functions modify enclosing vars:
>
> ```python
> def outer():
>     x = 0
>     def inner():
>         nonlocal x
>         x += 1
>     inner()
>     print(x)  # 1
> ```
>
> `global` lets you modify module-level variables:
>
> ```python
> x = 0
> def change():
>     global x
>     x += 1
> ```
>
> #### Danger:
> These aren’t just escape hatches—they mutate outer state. Think twice before using them in larger systems.

#### **55. Python has an opinionated standard library**  
Python ships batteries-included.  
`json`, `re`, `datetime`, `os`, `pathlib`, etc.—they’re all there.  
Learn to reach for these before installing third-party junk.

> [!info]- **Standard Library Cheatsheet**
> - `json` — encode/decode JSON  
>   ```python
>   import json
>   data = json.loads('{"x": 1}')
>   ```
> - `re` — regular expressions  
>   ```python
>   import re
>   re.findall(r'\d+', 'abc123')  # ['123']
>   ```
> - `datetime` — timestamps, deltas, parsing  
>   ```python
>   from datetime import datetime
>   now = datetime.now()
>   ```
> - `os` — environment, files, paths  
>   ```python
>   import os
>   os.listdir('.')  # list files in cwd
>   ```
> - `pathlib` — object-oriented file paths  
>   ```python
>   from pathlib import Path
>   config_path = Path.home() / ".config" / "my_app" / "settings.json"
>   if config_path.exists():
>       print(config_path.read_text())
>   ```
>> [!info] pathlib elaboration
>> Unlike `os.path`, `pathlib` treats file paths as **objects**, not strings.  
>> 
>> **Why it matters:**  
>> - **Joining paths** is clean:  
>>   ```python
>>   config = Path.home() / ".config" / "my_app"
>>   ```
>>   No `os.path.join()`, no fragile string hacks.  
>> - **File I/O is built-in**:  
>>   ```python
>>   content = config.read_text()
>>   config.write_text("hello")
>>   ```
>>   No `open()` boilerplate for common cases.  
>> - **Common checks are methods**:  
>>   `.exists()`, `.is_file()`, `.is_dir()`, `.glob("*.txt")`  
>> - **Cross-platform**: It just works—slashes, encodings, etc.  
>> - **C# comparison**: Think `Path.Combine` + `Directory` + `File`—but unified, cleaner, and chainable.  
>> 
>> `pathlib` is now the idiomatic standard. Only use `os.path` when forced.
> - `logging` — standard logging system, built-in  
>   ```python
>   import logging
>   logging.basicConfig(level=logging.INFO)
>   logger = logging.getLogger(__name__)
>   logger.info("This is an info message.")
>   ```
>> [!info] Logging anatomy
>> - **Loggers**: Created via `getLogger(name)`, usually `__name__`. They’re your interface for writing logs.  
>> - **Handlers**: Decide _where_ logs go—stdout, files, sockets, etc. Examples: `StreamHandler`, `FileHandler`.  
>> - **Formatters**: Control log message structure—timestamp, level, message, etc.  
>> - **basicConfig()**: Quick-and-dirty setup for small scripts. For apps, build a custom logger setup.  
>> - Think of it like Serilog or NLog in .NET, but already included. Learn it once—it scales.
> - `collections` — namedtuple, Counter, defaultdict  
>   ```python
>   from collections import Counter
>   Counter("hello")  # {'l': 2, 'h': 1, ...}
>   ```
> - `itertools` — iteration building blocks  
>   ```python
>   from itertools import cycle
>   c = cycle('AB')  # A B A B A B...
>   next(c), next(c)
>   ```
> - `functools` — higher-order function tools  
>   ```python
>   from functools import lru_cache, partial, wraps
>   ```
>   - `lru_cache`: memoize expensive calls. A simple decorator to add memoization (caching results of expensive function calls). 
>     ```python
>     @lru_cache
>     def fib(n):
>         return n if n < 2 else fib(n-1) + fib(n-2)
>     ```
>   - `partial`: fix some args ahead of time. Lets you "freeze" some of a function's arguments, creating a new function with a simplified signature. It's great for callbacks.  
>     ```python
>     from operator import mul
>     double = partial(mul, 2)
>     double(4)  # 8
>     ```
>   - `wraps`: preserve metadata when writing decorators. A helper for writing your own decorators. It ensures the wrapper function copies the metadata (like the name and docstring) from the original function, which is crucial for debugging.  
>     ```python
>     def logger(fn):
>         @wraps(fn)
>         def wrapper(*args, **kwargs):
>             print("calling", fn.__name__)
>             return fn(*args, **kwargs)
>         return wrapper
>     ```
>   These are Python’s built-in tools for higher-order control—great for decorators, caching, currying, or function composition.
> - `collections.defaultdict`: No more `if key not in dict` boilerplate  
>   ```python
>   from collections import defaultdict
>   d = defaultdict(list)
>   d["x"].append(1)
>   ```
> - `collections.Counter`: Count anything, fast  
>   ```python
>   from collections import Counter
>   Counter("bananas")  # {'a':3, 'b':1, 'n':2, 's':1}
>   ```
> - `argparse`: Build real CLI tools  
>   ```python
>   import argparse
>   parser = argparse.ArgumentParser()
>   parser.add_argument("--name")
>   args = parser.parse_args()
>   print(f"Hello, {args.name}")
>   ```
>
> These aren't trivia—they're **idiomatic tools**. Learn once, use forever.

> [!info]- **Beyond the Standard Library: Ecosystem Map**
> These are **not part of the standard library**, but are dominant in their domains:
>
> - **Web frameworks**
>   - `Django`: full-stack, batteries-included (like ASP.NET Core)
>   - `Flask`: minimalist and flexible
>   - `FastAPI`: built on Flask ideas, but fully type-hint driven
>
> - **Data validation & modeling**
>   - `Pydantic`: type-annotated classes with validation—like Newtonsoft.Json meets System.ComponentModel.DataAnnotations
>
> - **Database access**
>   - `SQLAlchemy Core`: low-level, expressive SQL abstraction (Dapper-esque)
>   - `SQLAlchemy ORM`: object-relational mapping (like Entity Framework)
>   - `Django ORM`: opinionated, declarative, powerful
>
> - **Data manipulation**
>   - `Pandas`: for tabular data, analysis, and CSV/Excel/database wrangling—Python’s LINQ + Excel + SQL
>
> - **HTTP clients**
>   - `requests`: like HttpClient but readable
>   - `httpx`: async-capable, drop-in requests replacement
>
> This is your "next steps" toolbox. Learn them when your project demands it—not before.

#### **56. Imports are absolute (mostly) and brittle (sometimes)**  
Python module resolution can get ugly in larger projects.  
Expect to deal with relative imports, `__init__.py`, and the occasional `sys.path` hack unless you use tooling or package layout cleanly.

> [!info]- **Import patterns in Python**
> 
> #### Absolute import (preferred):
> ```python
> from myproject.utils.helpers import do_the_thing
> ```
> This works as long as `myproject` is discoverable—i.e., in your `PYTHONPATH` or installed as a package.
> 
> #### Relative import:
> ```python
> from ..helpers import do_the_thing
> ```
> Useful within packages, but fragile if you run files directly (`python file.py` will break it).
> 
> **Fix: run modules properly**  
> Instead of `python file.py`, run it as a module to preserve relative imports:  
> ```bash
> python -m myproject.scripts.runner
> ```
> This sets up `__package__` correctly so relative imports don't explode.
> 
> #### `__init__.py` is required for package recognition:
> - Without it, directories aren’t considered packages.
> - Even empty `__init__.py` files signal to Python: “yes, treat this as a module.”
> 
> #### The nuclear option:
> ```python
> import sys
> sys.path.append('/some/path')
> ```
> Don’t do this unless you’re cornered. It mutates import paths globally and silently.
> 
> ---
> **Tip**: Use `python -m module` or `poetry run` / `pip install -e .` to keep imports sane.

#### **57. The import system is hookable**  
You can override how modules load using `importlib`, custom loaders, or even monkeypatching `sys.meta_path`. It’s horrifyingly powerful. You can import from zip files, URLs, or raw memory.

> [!info]- **Example: custom import hook**
> 
> Here’s a cursed demo that logs every module being imported:
> 
> ```python
> import sys
> 
> class Watcher:
>     def find_spec(self, name, path, target=None):
>         print(f"Importing: {name}")
>         return None  # Let normal import machinery continue
> 
> sys.meta_path.insert(0, Watcher())
> 
> import math
> import random
> ```
> 
> Outputs:
> ```
> Importing: math
> Importing: random
> ```
> 
> ---
> 
> You can go further:
> - Load code from encrypted blobs.  
> - Rewrite source before compile.
> - Stub/mock modules on import. 
> 
> Most people **shouldn’t** do this.  
> But if you must—`importlib` and `sys.meta_path` are your friends. Or demons.

> [!info]- **Further reading**
>
> You’re scratching the surface of Python's import internals. For the truly cursed path, look into AST manipulation, bytecode rewriting, and custom loaders. But bring gloves.

#### **58. `eval` and `exec` are live hand grenades**  
They run strings as Python code. That means **user input becomes executable**—and that’s almost never safe. Use in REPLs or controlled scripts only.

> [!info]- **`eval()` vs `exec()`**
>
> #### `eval()` — evaluates an expression and returns a value
> ```python
> x = 2
> result = eval("x + 3")  # 5
> ```
> Only works with **expressions** (not full statements).
>
> #### `exec()` — executes any Python code, no return
> ```python
> code = "for i in range(3): print(i)"
> exec(code)
> # 0
> # 1
> # 2
> ```
> Can run **any statement**, including control flow, function defs, imports.
>
> ---
>
> #### Why they’re dangerous:
> ```python
> user_input = "os.system('rm -rf /')"
> eval(user_input)  # goodbye system
> ```
> - If the string comes from a user, it can execute arbitrary code. 
> - No sandboxing, no checks, no mercy.
> - Equivalent to running pasted Python code into your terminal.
>
> ---
>
> #### Safe(ish) alternatives:
> - Use `ast.literal_eval()` for safe evaluation of literals (strings, numbers, lists, etc.)
>   ```python
>   from ast import literal_eval
>   literal_eval("[1, 2, 3]")  # ✅
>   literal_eval("__import__('os')")  # ❌ raises error
>   ```
>
> - For expression parsing or scripting, use controlled interpreters (like `lark`, `asteval`, etc.)
>
> ---
>
> Rule: if it runs code from a string, assume it's a trap.  
> Never `eval()` your way out of a real problem.

## Advanced Topics & Concurrency

#### **59. The GIL is real. Async is cooperative**.  
The Global Interpreter Lock makes true multithreading mostly a lie. Threads share memory, but only one runs Python bytecode at a time. Use `multiprocessing` for CPU-bound work. Use `asyncio` or `trio` for IO-bound tasks. Async/await isn't magic—it's cooperative multitasking. Coroutines yield control manually (`await`), allowing thousands of tasks to run _as long as none hogs the CPU_. No `await`, no multitasking.

> [!info]- **Explanation**
>
> Python’s **Global Interpreter Lock (GIL)** ensures only **one thread executes Python code at a time**, even on multicore CPUs. This kills true multithreading performance.
>
> ---
>
> #### Why does it exist?
> - Python (CPython) uses **reference counting** for memory management.
> - The GIL prevents memory corruption by avoiding race conditions.
> - It’s simpler, but it throttles threads to one-at-a-time bytecode execution.
> 
> (C extensions like NumPy can release the GIL—so parallelism isn’t dead, just not in pure Python.)
>
> ---
>
> #### CPU-bound? Use multiprocessing:
> ```python
> from multiprocessing import Process
>
> def work():
>     for _ in range(10**7):
>         pass
>
> p1 = Process(target=work)
> p2 = Process(target=work)
> p1.start()
> p2.start()
> p1.join()
> p2.join()
> ```
> - Each process has its **own GIL** and runs in parallel.
>
> ---
>
> #### IO-bound? Use async or threading:
> ```python
> import asyncio
>
> async def download():
>     await asyncio.sleep(1)
>     print("done")
>
> asyncio.run(download())
> ```
> - `await` **yields control** to the event loop.  
> - No blocking = thousands of tasks can be juggled.
>
> ---
>
> #### No await, no multitasking
> ```python
> async def bad():
>     time.sleep(1)  # blocks everything
>
> async def good():
>     await asyncio.sleep(1)  # yields
> ```
> - Coroutines must **cooperate**—the event loop doesn't preempt anything.
> - A single blocking call inside an `async def` will stall **every other task**.
> - Async isn't magic. It only works if **every task yields control voluntarily**.
>
> ---
> 
> #### Bonus
> - GIL is specific to **CPython**, the reference interpreter.  
>   Others (PyPy, Jython, IronPython) may not have it.
