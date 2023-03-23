set_project("cuda-path-tracer")

set_xmakever("2.7.5")

set_warnings("all")
set_languages("c++17")

add_rules("mode.debug", "mode.release")

add_requireconfs("*", {configs = {shared = true}})

if is_plat("windows") then
    if is_mode("debug") then
        set_runtimes("MDd")
    else
        set_runtimes("MT")
    end
end
-- support utf-8 on msvc
if is_host("windows") then
    add_defines("UNICODE", "_UNICODE")
    add_cxflags("/execution-charset:utf-8", "/source-charset:utf-8", {tools = "cl"})
end

add_requires("glew","glm", "stb","tinygltf")
add_requires("imgui", {configs = {glfw_opengl3 = true}})

-- dynamic link
add_links("cudart")

target("cuda_pt")
    set_kind("binary")
    add_files("src/*.cpp")
    add_files("src/*.cu")
    
    -- add_includedirs("external/include")
    

    add_packages("glew","glm", "stb", "imgui","tinygltf")

    set_rundir("$(projectdir)/scenes")
    set_runargs("scene.txt")

target_end()