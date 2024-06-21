import Printf
lines = readlines("/Users/sphw/Downloads/WMM2020COF/WMM2020COF/WMM.COF")
Printf.@printf("pub const WMM: &'static [(usize, usize, f64, f64, f64, f64)] = &[\n")
for line in lines[2:end]
    data = [x for x in split(line, " ") if x != ""]
    if data[1] == "999999999999999999999999999999999999999999999999"
        break
    end
    data = join(data, ", ")
    Printf.@printf("(%s),\n", data)
end
Printf.@printf("];\n")
