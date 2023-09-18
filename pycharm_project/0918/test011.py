import matplotlib.pyplot as plt

figsize = (7, 7)

fig, ax = plt.subplots(figsize=figsize)

# text alignment

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])

ax.grid()

# 예시
# ax.text(x=0, y=0, va='center', ha='left', s='Hello', fontsize=30)
# ax.text(x=0, y=0, va='center', ha='center', s='Hello', fontsize=30)
# ax.text(x=0, y=0, va='center', ha='right', s='Hello', fontsize=30)

# 실습
# ax.text(x=0, y=0, va='bottom', ha='right', s='Hello', fontsize=30)
# ax.text(x=0, y=0, va='bottom', ha='left', s='Hello', fontsize=30)
# ax.text(x=0, y=0, va='top', ha='right', s='Hello', fontsize=30)
ax.text(x=0, y=0, va='top', ha='left', s='Hello', fontsize=30)

plt.show()
