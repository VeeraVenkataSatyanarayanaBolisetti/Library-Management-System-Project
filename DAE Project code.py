import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from IPython.display import display

books = None

def show(data):
    if data.empty:
        print("No books found.")
    else:
        display(data.reset_index(drop=True))

def statuses():
    return books["status"].fillna("").astype(str).str.strip().str.lower()

def load_dataset():
    filename = "Books_data-Sheet1.csv"
    if not os.path.exists(filename):
        print("Upload your library CSV:")
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded.")
            return None
        filename = filename if filename in uploaded else next(iter(uploaded))
    try:
        data = pd.read_csv(filename)
        data.columns = data.columns.str.strip().str.lower()
        required = ["bid", "title", "author", "category", "status"]
        if not set(required).issubset(data.columns):
            raise ValueError("Required columns: " + ", ".join(required))
        if data.empty:
            raise ValueError("Dataset is empty.")
        ids = pd.to_numeric(data["bid"], errors="coerce")
        if ids.isna().any() or (ids % 1 != 0).any() or ids.duplicated().any():
            raise ValueError("Book IDs must be unique whole numbers.")
        data["bid"] = ids.astype(int)
        state = data["status"].fillna("").astype(str).str.strip().str.lower()
        if not state.isin(["available", "issued"]).all():
            raise ValueError("Status must be available or issued.")
        rng = np.random.default_rng(42)
        added = []
        if "borrow_count" not in data:
            data["borrow_count"] = rng.integers(0, 51, len(data))
            added.append("borrow_count")
        if "fine" not in data:
            data["fine"] = rng.choice([0, 10, 20, 30, 50, 100], len(data))
            added.append("fine")
        for col in ["borrow_count", "fine"]:
            values = pd.to_numeric(data[col], errors="coerce")
            if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{col} must contain finite, non-negative numbers.")
            data[col] = values
        if (data["borrow_count"] % 1 != 0).any():
            raise ValueError("Borrow counts must be whole numbers.")
        data["borrow_count"] = data["borrow_count"].astype(int)
        print(f"Loaded {len(data)} books.")
        if added:
            print("Sample columns added:", ", ".join(added))
            print("These are demonstration values, not actual records.")
        show(data.head())
        return data
    except Exception as error:
        print("Load error:", error)
        return None

def find_book():
    try:
        book_id = int(input("Book ID: "))
    except ValueError:
        print("Enter a numeric Book ID.")
        return None
    matches = books.index[books["bid"] == book_id]
    if len(matches) == 0:
        print("Book ID not found.")
        return None
    return matches[0]

def manage_books():
    global books
    while True:
        print("""
--- MANAGE BOOKS ---
1. Search Books
2. View Available Books
3. Issue Book
4. Return Book
5. Add Book
6. Back
""")
        choice = input("Choice: ").strip()
        if choice == "6":
            break
        elif choice == "1":
            term = input("Title, author, or category: ").strip()
            if not term:
                print("Enter a search term.")
                continue
            mask = pd.Series(False, index=books.index)
            for col in ["title", "author", "category"]:
                mask |= books[col].fillna("").astype(str).str.contains(
                    term, case=False, regex=False
                )
            show(books[mask])
        elif choice == "2":
            show(books[statuses() == "available"])
        elif choice in ["3", "4"]:
            index = find_book()
            if index is None:
                continue
            target = "issued" if choice == "3" else "available"
            if statuses().loc[index] == target:
                print(f"Book is already {target}.")
                continue
            books.at[index, "status"] = target
            if choice == "3":
                books.at[index, "borrow_count"] += 1
            print(f'{books.at[index, "title"]}: {target}')
            if choice == "4":
                print("Fine unchanged; due dates are not tracked.")
        elif choice == "5":
            title = input("Title: ").strip()
            author = input("Author: ").strip()
            category = input("Category: ").strip()
            if not all([title, author, category]):
                print("All fields are required.")
                continue
            new_id = int(books["bid"].max()) + 1
            record = dict(bid=new_id, title=title, author=author,
                          category=category, status="available",
                          borrow_count=0, fine=0)
            books = pd.concat([books, pd.DataFrame([record])], ignore_index=True)
            print(f"Book added. ID: {new_id}")
        else:
            print("Invalid choice.")

def dataset_info():
    print(f"Rows: {books.shape[0]}, Columns: {books.shape[1]}")
    books.info()
    print("\nMissing values:")
    display(books.isnull().sum().rename("Missing Values").to_frame())
    print("Duplicate rows:", books.duplicated().sum())

def numpy_analysis():
    borrow = books["borrow_count"].to_numpy()
    fine = books["fine"].to_numpy()
    print("\n--- Borrowing Statistics ---")
    for label, function in [
        ("Total", np.sum), ("Mean", np.mean), ("Median", np.median),
        ("Minimum", np.min), ("Maximum", np.max), ("Standard deviation", np.std)
    ]:
        print(f"{label}: {function(borrow):.2f}")
    print("\n--- Fine Statistics ---")
    print(f"Total: Rs. {np.sum(fine):.2f}")
    print(f"Average: Rs. {np.mean(fine):.2f}")
    print(f"Maximum: Rs. {np.max(fine):.2f}")

def pandas_analysis():
    print("\nNumerical summary:")
    display(books[["borrow_count", "fine"]].describe())
    print("\nCategory-wise analysis:")
    display(books.groupby("category", dropna=False).agg(
        total_books=("bid", "size"),
        total_borrowings=("borrow_count", "sum"),
        average_borrowings=("borrow_count", "mean"),
        total_fine=("fine", "sum")
    ).round(2))
    print("\nStatus counts:")
    display(statuses().value_counts().rename("Book Count").to_frame())
    print("\nTop 5 most borrowed books:")
    show(books.nlargest(5, "borrow_count")[
        ["bid", "title", "author", "borrow_count", "status"]
    ])

def draw_chart(kind):
    counts = books["category"].fillna("Unknown").value_counts()
    if kind == "5":
        state = statuses().to_numpy()
        available = int(np.count_nonzero(state == "available"))
        issued = int(np.count_nonzero(state == "issued"))
        print(f"Total: {len(books)}, Available: {available}, Issued: {issued}")
        print(f"Availability: {available / len(books):.1%}")
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes[0].bar(["Available", "Issued"], [available, issued],
                    color=["seagreen", "tomato"])
        axes[0].set(title="Book Availability", ylabel="Number of Books")
        counts.sort_values().plot.barh(ax=axes[1], color="steelblue")
        axes[1].set(title="Books by Category", xlabel="Number of Books")
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        if kind == "1":
            ax.hist(books["borrow_count"], bins=10,
                    color="skyblue", edgecolor="black")
            ax.set(title="Book Borrowing Histogram",
                   xlabel="Borrow Count per Book", ylabel="Number of Books")
        elif kind == "2":
            counts.plot.bar(ax=ax, color="teal", edgecolor="black")
            ax.set(title="Book Category Bar Chart",
                   xlabel="Category", ylabel="Number of Books")
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
        elif kind == "3":
            ax.scatter(books["borrow_count"], books["fine"],
                       color="purple", alpha=0.65, edgecolors="black")
            ax.set(title="Borrow Count vs Fine",
                   xlabel="Borrow Count per Book", ylabel="Fine per Book (Rs.)")
            ax.grid(alpha=0.3)
            print("Sample values do not establish a real borrowing–fine relationship.")
        elif kind == "4":
            state = statuses().value_counts()
            ax.pie(state.values, labels=state.index,
                   autopct="%1.1f%%", startangle=90)
            ax.set_title("Book Status Pie Chart")
            ax.axis("equal")
    plt.tight_layout()
    plt.show()

def charts_menu():
    while True:
        print("""
--- CHARTS AND STATISTICS ---
1. Borrowing Histogram
2. Category Bar Chart
3. Borrow Count vs Fine Scatter Plot
4. Book Status Pie Chart
5. Combined Statistics and Charts
6. Show All Charts
7. Back
""")
        choice = input("Choice: ").strip()
        if choice == "7":
            break
        elif choice == "6":
            for kind in ["1", "2", "3", "4", "5"]:
                draw_chart(kind)
        elif choice in ["1", "2", "3", "4", "5"]:
            draw_chart(choice)
        else:
            print("Invalid choice.")

def recommend_books():
    print("Categories:", ", ".join(sorted(
        books["category"].dropna().astype(str).unique()
    )))
    category = input("Category (Enter for all): ").strip()
    result = books[statuses() == "available"]
    if category:
        mask = result["category"].fillna("").astype(str).str.strip().str.casefold()
        result = result[mask == category.casefold()]
    result = result.sort_values(
        ["borrow_count", "bid"], ascending=[False, True]
    ).head(5)
    print("Available books ranked by borrow count.")
    print("Sample counts produce demonstration rankings.")
    show(result[["bid", "title", "author", "category", "borrow_count", "status"]])

while True:
    print("""
================================================
   LIBRARY MANAGEMENT AND DATA ANALYSIS SYSTEM
================================================
1. Create / Load Dataset
2. Display All Books
3. Manage Books
4. Dataset Information
5. NumPy Statistical Analysis
6. Pandas Data Analysis
7. Charts and Statistics
8. Book Recommendation
9. Download Updated Dataset
10. Exit
================================================
""")
    choice = input("Enter your choice (1-10): ").strip()
    if choice == "10":
        print("Session ended. Only downloaded changes are saved outside Colab.")
        break
    elif choice == "1":
        if books is not None:
            if input("Reload and discard current changes? (yes/no): ").strip().lower() != "yes":
                continue
        loaded = load_dataset()
        if loaded is not None:
            books = loaded
    elif choice not in [str(i) for i in range(2, 10)]:
        print("Invalid choice.")
    elif books is None:
        print("Select option 1 first.")
    elif choice == "9":
        books.to_csv("updated_library_books.csv", index=False)
        files.download("updated_library_books.csv")
        print("Download requested. Original CSV is not overwritten.")
    else:
        actions = {"2": lambda: show(books), "3": manage_books,
                   "4": dataset_info, "5": numpy_analysis,
                   "6": pandas_analysis, "7": charts_menu,
                   "8": recommend_books}
        actions[choice]()