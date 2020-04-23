# Add google test to existing project
find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIRS})

macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    # It is important to put pthread as target link library as last entry.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main Cassemble pthread)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    #set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    #set_property(TARGET ${TESTNAME} PROPERTY CXX_STANDARD 17)
endmacro()

package_add_test(dummy assembleTest.cpp)