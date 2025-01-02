import mcschematic
schem = mcschematic.MCSchematic()
def number_command_block(num,x,y,z):
    input_number=str(int(2**num))
    block = "minecraft:repeating_command_block{SuccessCount:" + input_number + "}"
    schem.setBlock((x, y, z), block)
for i in range(0,31):
    number_command_block(i,0-4*i,0,0)
    number_command_block(i,-1-4*i,0,-2)
schem.save("myschems", "Weights3", mcschematic.Version.JE_1_18_2)
schem = mcschematic.MCSchematic()
for i in range(0,24):
    number_command_block(i,0-4*i,0,0)
    number_command_block(i,-1-4*i,0,-2)
schem.save("myschems", "Weights4", mcschematic.Version.JE_1_18_2)