import pandas as pd

# Load sheets
file = "Microfinance_BI_Dataset.xlsx"

customers = pd.read_excel(file, sheet_name="Customers")
loans = pd.read_excel(file, sheet_name="Loans")
repayments = pd.read_excel(file, sheet_name="Repayments")
officers = pd.read_excel(file, sheet_name="Loan_Officers")

print("Sheets loaded...")

# Merge customer details
master = pd.merge(loans, customers, on="CustomerID", how="left")

# Merge officer name
master = pd.merge(master, officers[["OfficerID", "OfficerName"]], on="OfficerID", how="left")

# Merge repayment details
rep_group = repayments.groupby("LoanID", as_index=False).agg({
    "AmountPaid": "sum",
    "DueAmount": "sum",
    "DelayDays": "max",
    "PaymentDate": "max"
})

master = pd.merge(master, rep_group, on="LoanID", how="left")

# Save final CSV
master.to_csv("Microfinance_Final_Loan_Master.csv", index=False)

print("================================")
print("CSV CREATED SUCCESSFULLY")
print("Microfinance_Final_Loan_Master.csv")
print("================================")