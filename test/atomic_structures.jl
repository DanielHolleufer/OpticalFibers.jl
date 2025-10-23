using Test
using OpticalFibers

@testset "NLevelAtom isequal and hash" begin
    γ_1 = 1.0
    d_1 = [1.0, 0.0, 0.0]
    two_level_atom_1 = TwoLevelAtom(γ_1, d_1)

    γ_2 = 1.0
    d_2 = [1.0, 0.0, 0.0]
    two_level_atom_2 = TwoLevelAtom(γ_2, d_2)

    γ_3 = 2.0
    d_3 = [1.0, 0.0, 0.0]
    two_level_atom_3 = TwoLevelAtom(γ_3, d_3)

    γ_4 = 1.0
    d_4 = [0.0, 1.0, 0.0]
    two_level_atom_4 = TwoLevelAtom(γ_4, d_4)
    
    @test isequal(two_level_atom_1, two_level_atom_2)
    @test isequal(hash(two_level_atom_1), hash(two_level_atom_2))

    @test !isequal(two_level_atom_1, two_level_atom_3)
    @test !isequal(two_level_atom_1, two_level_atom_4)
    @test !isequal(hash(two_level_atom_1), hash(two_level_atom_3))
    @test !isequal(hash(two_level_atom_1), hash(two_level_atom_4))

    Δ_1 = 0.0
    γ_lower_1 = 1.0
    γ_upper_1 = 0.1
    d_lower_1 = [1.0, 0.0, 0.0]
    d_upper_1 = [0.0, 1.0, 0.0]
    three_level_atom_1 = ThreeLevelAtom(Δ_1, γ_lower_1, γ_upper_1, d_lower_1, d_upper_1)

    Δ_2 = 0.0
    γ_lower_2 = 1.0
    γ_upper_2 = 0.1
    d_lower_2 = [1.0, 0.0, 0.0]
    d_upper_2 = [0.0, 1.0, 0.0]
    three_level_atom_2 = ThreeLevelAtom(Δ_2, γ_lower_2, γ_upper_2, d_lower_2, d_upper_2)

    Δ_3 = 1.0
    γ_lower_3 = 1.0
    γ_upper_3 = 0.1
    d_lower_3 = [1.0, 0.0, 0.0]
    d_upper_3 = [0.0, 1.0, 0.0]
    three_level_atom_3 = ThreeLevelAtom(Δ_3, γ_lower_3, γ_upper_3, d_lower_3, d_upper_3)

    Δ_4 = 0.0
    γ_lower_4 = 2.0
    γ_upper_4 = 0.1
    d_lower_4 = [1.0, 0.0, 0.0]
    d_upper_4 = [0.0, 1.0, 0.0]
    three_level_atom_4 = ThreeLevelAtom(Δ_4, γ_lower_4, γ_upper_4, d_lower_4, d_upper_4)

    Δ_5 = 0.0
    γ_lower_5 = 1.0
    γ_upper_5 = 0.5
    d_lower_5 = [1.0, 0.0, 0.0]
    d_upper_5 = [0.0, 1.0, 0.0]
    three_level_atom_5 = ThreeLevelAtom(Δ_5, γ_lower_5, γ_upper_5, d_lower_5, d_upper_5)

    Δ_6 = 0.0
    γ_lower_6 = 1.0
    γ_upper_6 = 0.1
    d_lower_6 = [0.0, 1.0, 0.0]
    d_upper_6 = [0.0, 1.0, 0.0]
    three_level_atom_6 = ThreeLevelAtom(Δ_6, γ_lower_6, γ_upper_6, d_lower_6, d_upper_6)

    Δ_7 = 0.0
    γ_lower_7 = 1.0
    γ_upper_7 = 0.1
    d_lower_7 = [1.0, 0.0, 0.0]
    d_upper_7 = [0.0, -1.0, 0.0]
    three_level_atom_7 = ThreeLevelAtom(Δ_7, γ_lower_7, γ_upper_7, d_lower_7, d_upper_7)

    @test isequal(three_level_atom_1, three_level_atom_2)
    @test isequal(hash(three_level_atom_1), hash(three_level_atom_2))

    @test !isequal(three_level_atom_1, three_level_atom_3)
    @test !isequal(three_level_atom_1, three_level_atom_4)
    @test !isequal(three_level_atom_1, three_level_atom_5)
    @test !isequal(three_level_atom_1, three_level_atom_6)
    @test !isequal(three_level_atom_1, three_level_atom_7)
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_3))
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_4))
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_5))
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_6))
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_7))
end
