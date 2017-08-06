
a={torch.ones(2)}

b={}

for i=1,2 do
   a[1]=torch.ones(2)
   table.insert(b, a[1])
end

a[1][1]=2
print(b[1])
print(b[2])
b[1][1]=3
print(b[1])
print(b[2])

