#!/bin/bash
# Predict reimbursement using a polynomial regression model trained on the
# public dataset. The script expects three numeric parameters:
#   $1 - trip duration in days
#   $2 - miles traveled
#   $3 - total receipts amount
python3 - "$1" "$2" "$3" <<'PY'
import sys
try:
    d=float(sys.argv[1])
    m=float(sys.argv[2])
    r=float(sys.argv[3])
except Exception:
    print("0")
    sys.exit(0)
coefs=[0.0,135.09738794978105,0.08854921822588165,0.6081843213591666,-13.055627769510634,0.046058273285780514,0.0012117847617202082,0.00041241927422318894,0.0002458712596111166,0.00017387800345329202,0.5407011200054512,-0.004732764835287777,0.0009654478844031088,4.412408925750146e-05,-1.4466658311641778e-05,-6.318838194186628e-06,-3.6880328084305603e-07,-4.792942211934693e-08,-8.770365903317141e-08,-9.59149847919476e-08]
intercept=-56.35660932310748
features=[1,d,m,r,d**2,d*m,d*r,m**2,m*r,r**2,d**3,d**2*m,d**2*r,d*m**2,d*m*r,d*r**2,m**3,m**2*r,m*r**2,r**3]
pred=intercept+sum(c*f for c,f in zip(coefs,features))
print(f"{pred:.2f}")
PY
